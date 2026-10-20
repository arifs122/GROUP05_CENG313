.PHONY: help setup check arena inspect test tournament tournament-bo3 submission submission-bot clean

BOT ?= bots/_template

help:
	@echo "setup          : copy cg/ and card data into git-ignored folders"
	@echo "check          : verify the local simulator"
	@echo "arena          : 20 games, heuristic vs random"
	@echo "inspect        : print 15 observations"
	@echo "test           : run smoke tests"
	@echo "tournament     : round-robin, 50 matches per pair"
	@echo "tournament-bo3 : round-robin, 30 BO3 matches per pair"
	@echo "submission     : build submissions/submission.tar.gz from src/agent"
	@echo "submission-bot : build a bot bundle (BOT=bots/name)"
	@echo "clean          : remove build artifacts"

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

tournament:
	python3 tools/tournament.py --games 50

tournament-bo3:
	python3 tools/tournament.py --games 30 --bo3

submission:
	python3 tools/build_submission.py

submission-bot:
	python3 tools/build_submission.py --bot $(BOT)

clean:
	rm -rf build submissions
