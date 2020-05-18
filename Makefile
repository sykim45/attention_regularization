cifar100_pyramidnet200_high_reconstruct_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type pyramidnet \
		--model reconstruct \
		--dropout_type high \
		--depth 200 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_pyramidnet200_cutmix_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type pyramidnet \
		--model reconstruct \
		--dropout_type basic \
		--depth 200 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_pyramidnet200_noise_reconstruct_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type pyramidnet \
		--model reconstruct \
		--dropout_type noise \
		--depth 200 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_pyramidnet200_high_reconstruct_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type pyramidnet \
		--model reconstruct \
		--dropout_type high \
		--depth 200 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_pyramidnet200_cutmix_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type pyramidnet \
		--model reconstruct \
		--dropout_type basic \
		--depth 200 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_pyramidnet200_noise_reconstruct_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type pyramidnet \
		--model reconstruct \
		--dropout_type noise \
		--depth 200 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly


cifar100_resnet18_train:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model vanilla \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet18_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model vanilla \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet18_basic_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type basic \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet18_basic_test:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type basic \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet18_att_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type attention \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet18_att_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type attention \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet18_att_reconstruct_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type attention \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--write_plot True \
		--dataset cifar100

cifar100_resnet18_att_reconstruct_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type attention \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet18_random_reconstruct_train:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type random \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--write_plot True \
		--dataset cifar100

cifar100_resnet18_random_reconstruct_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type random \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly


cifar100_resnet18_high_reconstruct_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type high \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet18_high_reconstruct_test:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type high \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly


cifar100_resnet18_cutmix_train:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type basic \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet18_cutmix_test:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type basic \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet50_train:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model vanilla \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet18_cutmix_dropout_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type basicdropout \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet18_cutmix_dropout_test:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type basicdropout \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly


cifar100_resnet50_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model vanilla \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet50_basic_train:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type basic \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet50_basic_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type basic \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet50_att_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type attention \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet50_att_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type attention \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet50_att_reconstruct_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type attention \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet50_att_reconstruct_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type attention \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly


cifar100_resnet152_train:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model vanilla \
		--depth 152 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet152_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model vanilla \
		--depth 152 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet152_basic_train:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type basic \
		--depth 152 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet152_basic_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type basic \
		--depth 152 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet152_att_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type attention \
		--depth 152 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet152_att_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type attention \
		--depth 152 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet152_att_reconstruct_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type attention \
		--depth 152 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet152_att_reconstruct_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type attention \
		--depth 152 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet50_cutmix_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type basic \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet50_cutmix_test:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type basic \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

cifar100_resnet152_cutmix_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type basic \
		--depth 152 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100

cifar100_resnet152_cutmix_test:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type basic \
		--depth 152 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset cifar100 \
		--testOnly

esnet18_train:
	python src/main.py \
		--net_type resnet \
		--model vanilla \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset tiny_imagenet

resnet18_test:
	python src/main.py \
		--net_type resnet \
		--model vanilla \
		--depth 18 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset tiny_imagenet \
		--testOnly

resnet50_msr_train:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model vanilla \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset tiny_imagenet

resnet50_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model vanilla \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset tiny_imagenet \
		--testOnly

resnet50_att_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type attention \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset tiny_imagenet

resnet50_att_reconstruct_train:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type attention \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset tiny_imagenet

resnet50_att_test:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type attention \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset tiny_imagenet \
		--testOnly

resnet50_att_reconstruct_test:
	CUDA_VISIBLE_DEVICES='1' \
	python src/main.py \
		--net_type resnet \
		--model reconstruct \
		--dropout_type attention \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset tiny_imagenet \
		--testOnly

resnet50_basic_train:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type basic \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset tiny_imagenet

resnet50_basic_test:
	CUDA_VISIBLE_DEVICES='0' \
	python src/main.py \
		--net_type resnet \
		--model dropout \
		--dropout_type basic \
		--depth 50 \
		--conv_init MSR \
		--fc_init kaiming \
		--dataset tiny_imagenet \
		--testOnly