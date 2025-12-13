import numpy as np
import ctypes
import time
import argparse
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from OpenGL.EGL import *
from OpenGL.EGL.EXT.platform_base import *

# Load OpenGL library for manual function loading
try:
    libGL = ctypes.CDLL("libGL.so.1")
except:
    libGL = ctypes.CDLL("libGL.so")

# Define compute shader functions manually
_glDispatchCompute = None
_glMemoryBarrier = None

def load_gl_functions():
    """Load OpenGL 4.3+ compute shader functions manually"""
    global _glDispatchCompute, _glMemoryBarrier
    
    # Get function pointers
    _glDispatchCompute = libGL.glDispatchCompute
    _glDispatchCompute.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
    _glDispatchCompute.restype = None
    
    _glMemoryBarrier = libGL.glMemoryBarrier
    _glMemoryBarrier.argtypes = [ctypes.c_uint]
    _glMemoryBarrier.restype = None

def ensure_context():
    """Create an EGL context for headless GPU computing"""
    # Get EGL display
    egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
    if egl_display == EGL_NO_DISPLAY:
        raise RuntimeError("Failed to get EGL display")
    
    # Initialize EGL
    major, minor = ctypes.c_long(), ctypes.c_long()
    if not eglInitialize(egl_display, ctypes.pointer(major), ctypes.pointer(minor)):
        raise RuntimeError("Failed to initialize EGL")
    
    print(f"EGL version: {major.value}.{minor.value}")
    
    # Choose config
    config_attribs = [
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    ]
    config_attribs_array = (ctypes.c_int * len(config_attribs))(*config_attribs)
    
    num_configs = ctypes.c_long()
    configs = (EGLConfig * 1)()
    
    if not eglChooseConfig(egl_display, config_attribs_array, configs, 1, ctypes.pointer(num_configs)):
        raise RuntimeError("Failed to choose EGL config")
    
    if num_configs.value == 0:
        raise RuntimeError("No suitable EGL configs found")
    
    # Bind OpenGL API
    if not eglBindAPI(EGL_OPENGL_API):
        raise RuntimeError("Failed to bind OpenGL API")
    
    # Create context
    context_attribs = [EGL_NONE]
    context_attribs_array = (ctypes.c_int * len(context_attribs))(*context_attribs)
    
    egl_context = eglCreateContext(egl_display, configs[0], EGL_NO_CONTEXT, context_attribs_array)
    if egl_context == EGL_NO_CONTEXT:
        raise RuntimeError("Failed to create EGL context")
    
    # Create a pbuffer surface (since we're headless)
    pbuffer_attribs = [
        EGL_WIDTH, 1,
        EGL_HEIGHT, 1,
        EGL_NONE
    ]
    pbuffer_attribs_array = (ctypes.c_int * len(pbuffer_attribs))(*pbuffer_attribs)
    
    egl_surface = eglCreatePbufferSurface(egl_display, configs[0], pbuffer_attribs_array)
    if egl_surface == EGL_NO_SURFACE:
        raise RuntimeError("Failed to create EGL surface")
    
    # Make context current
    if not eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context):
        raise RuntimeError("Failed to make EGL context current")
    
    # Check OpenGL version
    gl_version = glGetString(GL_VERSION)
    gl_renderer = glGetString(GL_RENDERER)
    print(f"OpenGL Version: {gl_version.decode() if gl_version else 'Unknown'}")
    print(f"OpenGL Renderer: {gl_renderer.decode() if gl_renderer else 'Unknown'}")
    
    # Load compute shader functions
    load_gl_functions()
    
    return egl_display, egl_surface, egl_context

def cleanup_context(egl_display, egl_surface, egl_context):
    """Clean up EGL context"""
    eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
    if egl_context != EGL_NO_CONTEXT:
        eglDestroyContext(egl_display, egl_context)
    if egl_surface != EGL_NO_SURFACE:
        eglDestroySurface(egl_display, egl_surface)
    eglTerminate(egl_display)

def create_ssbo_from_numpy(binding, np_array, usage=GL_STATIC_DRAW):
    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    size = np_array.nbytes
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, None, usage)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, size, ctypes.c_void_p(np_array.ctypes.data))
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbo)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return ssbo

def create_ssbo_reserve(binding, size_bytes, usage=GL_DYNAMIC_READ):
    ssbo = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, size_bytes, None, usage)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbo)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return ssbo

def read_ssbo_to_numpy(ssbo, nbytes, dtype=np.float32, count=None):
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    buf = ctypes.create_string_buffer(nbytes)
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, nbytes, buf)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    if count is None:
        count = nbytes // np.dtype(dtype).itemsize
    arr = np.frombuffer(buf, dtype=dtype, count=count).copy()
    return arr

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["naive", "chunked", "strassen"], default="naive")
parser.add_argument("--size", choices=["small", "medium", "big", "huge", "massive"], default="medium")
args = parser.parse_args()

mode = args.mode
N = 0
if args.size=="small":
    N=128
elif args.size=="medium":
    N=1024
elif args.size=="big":
    N=8192
elif args.size=="huge":
    N=16384
elif args.size=="massive":
    N=32768
TILE = 16
BLOCK_ROWS = 512  

shader_file = f"matmul_{mode}.comp"
with open(shader_file, "r") as f:
    src = f.read()

egl_display, egl_surface, egl_context = ensure_context()

try:
    prog = compileProgram(compileShader(src, GL_COMPUTE_SHADER))
    glUseProgram(prog)

    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.empty((N, N), dtype=np.float32)

    if mode == "chunked" or mode == "strassen":
        Npad = N if N % TILE == 0 else ((N + TILE - 1) // TILE) * TILE
        Apad = np.zeros((Npad, Npad), dtype=np.float32)
        Bpad = np.zeros((Npad, Npad), dtype=np.float32)
        Apad[:N,:N] = A
        Bpad[:N,:N] = B
        Cpad = np.empty((Npad, Npad), dtype=np.float32)
    else:
        Npad = N
        Apad = A
        Bpad = B
        Cpad = C

    ssbo_A = create_ssbo_from_numpy(0, Apad)
    ssbo_B = create_ssbo_from_numpy(1, Bpad)
    ssbo_C = create_ssbo_reserve(2, Cpad.nbytes)

    locN = glGetUniformLocation(prog, "N")
    if locN != -1: glUniform1i(locN, N)
    locStride = glGetUniformLocation(prog, "stride")
    if locStride != -1: glUniform1i(locStride, Npad)
    locBase = glGetUniformLocation(prog, "baseRow")
    locOffsetA = glGetUniformLocation(prog, "offsetA")
    locOffsetB = glGetUniformLocation(prog, "offsetB")

    start_time = time.time()

    if mode == "naive":
        groups_x = (N + TILE - 1) // TILE
        groups_y = (N + TILE - 1) // TILE
        _glDispatchCompute(groups_x, groups_y, 1)
        _glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glFinish()
    elif mode == "strassen":
        # Simple strassen - just use tiled multiply for now
        # Full Strassen recursion would require multiple passes
        groups_x = (Npad + TILE - 1) // TILE
        groups_y = (Npad + TILE - 1) // TILE
        if locOffsetA != -1: glUniform1i(locOffsetA, 0)
        if locOffsetB != -1: glUniform1i(locOffsetB, 0)
        _glDispatchCompute(groups_x, groups_y, 1)
        _glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glFinish()
    else:  # chunked
        groups_x = (Npad + TILE - 1) // TILE
        for base in range(0, N, BLOCK_ROWS):
            rows_this = min(BLOCK_ROWS, N - base)
            groups_y = (rows_this + TILE - 1) // TILE
            glUseProgram(prog)
            glUniform1i(locBase, base)
            _glDispatchCompute(groups_x, groups_y, 1)
            _glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glFinish()

    end_time = time.time()
    gpu_time = (end_time - start_time)*1000.0
    print(f"Matrix size: {N}x{N}, shader={mode}")
    print(f"GPU walltime = {gpu_time:.3f} ms")

    arr = read_ssbo_to_numpy(ssbo_C, Cpad.nbytes, count=Npad*Npad)
    Cpad_gpu = arr.reshape((Npad, Npad))
    C_gpu = Cpad_gpu[:N, :N]   

    samples = [(0,0),(N//2,N//3),(N-1,N-1)]
    for i,j in samples:
        gpu_val = float(C_gpu[i,j])
        cpu_val = float(np.dot(A[i,:], B[:,j]))
        print(f"sample ({i},{j}): GPU={gpu_val:.6f}, CPU={cpu_val:.6f}, diff={abs(gpu_val-cpu_val):.6f}")

    glDeleteBuffers(3, (GLuint*3)(ssbo_A, ssbo_B, ssbo_C))
    glDeleteProgram(prog)

finally:
    cleanup_context(egl_display, egl_surface, egl_context)
