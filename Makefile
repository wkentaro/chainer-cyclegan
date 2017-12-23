all:
	@echo '## Make commands ##'
	@echo
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$' | xargs

install:
	.make/install.sh

pytorch2chainer: install
	.make/pytorch2chainer.sh

test_pytorch2chainer: pytorch2chainer
	.make/test_pytorch2chainer.sh
