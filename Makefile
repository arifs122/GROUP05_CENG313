.PHONY: help setup check arena inspect test submission clean

help:
	@echo "setup      : copy cg/ and card data into git-ignored folders"
	@echo "check      : verify the local simulator"
	@echo "arena      : 20 games, heuristic vs random"
	@echo "inspect    : print 15 observations"
	@echo "test       : run smoke tests"
	@echo "submission : build submissions/submission.tar.gz"
	@echo "clean      : remove build artifacts"

setup:
	bash scripts/setup_assets.sh

check:
	python3 scripts/check_env.py

arena:
	python3 tools/arena.py --p0 heuristic --p1 random --games 20

inspect:
	python3 tools/inspect_obs.py --steps 15

test:
	python3 -m unittest discover -s tests -v

submission:
	python3 tools/build_submission.py

clean:
	rm -rf build submissions
