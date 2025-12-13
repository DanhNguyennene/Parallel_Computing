.PHONY: all_tests \
        test_naive_small test_naive_medium test_naive_big test_naive_huge test_naive_massive \
        test_chunked_small test_chunked_medium test_chunked_big test_chunked_huge test_chunked_massive \
        test_strassen_small test_strassen_medium test_strassen_big test_strassen_huge

test_naive_small:
	@echo "===== Running Naive Shader: Small Matrix ====="
	cd shader && python main_headless.py --mode naive --size small
	@echo ""

test_naive_medium:
	@echo "===== Running Naive Shader: Medium Matrix ====="
	cd shader && python main_headless.py --mode naive --size medium
	@echo ""

test_naive_big:
	@echo "===== Running Naive Shader: Big Matrix ====="
	cd shader && python main_headless.py --mode naive --size big
	@echo ""

test_chunked_small:
	@echo "===== Running Chunked Shader: Small Matrix ====="
	cd shader && python main_headless.py --mode chunked --size small
	@echo ""

test_chunked_medium:
	@echo "===== Running Chunked Shader: Medium Matrix ====="
	cd shader && python main_headless.py --mode chunked --size medium
	@echo ""

test_chunked_big:
	@echo "===== Running Chunked Shader: Big Matrix ====="
	cd shader && python main_headless.py --mode chunked --size big
	@echo ""

test_naive_huge:
	@echo "===== Running Naive Shader: Huge Matrix (16384x16384) ====="
	cd shader && python main_headless.py --mode naive --size huge
	@echo ""

test_chunked_huge:
	@echo "===== Running Chunked Shader: Huge Matrix (16384x16384) ====="
	cd shader && python main_headless.py --mode chunked --size huge
	@echo ""

test_naive_massive:
	@echo "===== Running Naive Shader: Massive Matrix (32768x32768) ====="
	cd shader && python main_headless.py --mode naive --size massive
	@echo ""

test_chunked_massive:
	@echo "===== Running Chunked Shader: Massive Matrix (32768x32768) ====="
	cd shader && python main_headless.py --mode chunked --size massive
	@echo ""

test_strassen_small:
	@echo "===== Running Strassen Shader: Small Matrix ====="
	cd shader && python main_headless.py --mode strassen --size small
	@echo ""

test_strassen_medium:
	@echo "===== Running Strassen Shader: Medium Matrix ====="
	cd shader && python main_headless.py --mode strassen --size medium
	@echo ""

test_strassen_big:
	@echo "===== Running Strassen Shader: Big Matrix ====="
	cd shader && python main_headless.py --mode strassen --size big
	@echo ""

test_strassen_huge:
	@echo "===== Running Strassen Shader: Huge Matrix ====="
	cd shader && python main_headless.py --mode strassen --size huge
	@echo ""

all_tests: \
	test_naive_small \
	test_naive_medium \
	test_naive_big \
	test_naive_huge \
	test_chunked_small \
	test_chunked_medium \
	test_chunked_big \
	test_chunked_huge \
	test_strassen_small \
	test_strassen_medium \
	test_strassen_big \
	test_strassen_huge
	@echo "===== All tests completed ====="
