# 病理图预测TMB/MSI等基因指标

## To-Do

- [ ] 要不要打包成docker(考虑环境和别的问题)
- [ ] 把预处理流程打包在一起

## 结果

|方法|结果|{batch_size}\_{lr}\_{Epoch}|{ACC}|{Recall}|{F1}|{AUC}|
|--------|--------|--------|--------|--------|--------|--------|
|tile|
|resnet18|不好|config={}||


## 1. 预处理步骤

>  * `执行命令`: ```python utils/slide_core.py```
>  * `参数设置`：
>       * `-m`/`--magnification`: 选择病理图的放大倍数
>       * `--tile_size`: 选择切割出来的tile的大小，常用512
>       * `--document_root`: 
>       * `--data_root`: 存放数据的根目录


## 2. data preparing

1. csv合并

    - [x] 合并tile的信息
    - [x] 合并slides和tmb的信息
    - [x] 分train-test
    
2. 建立dataset
   
    - [x] 基于tile建立dataset
    - [x] 基于mil建立dataset

3. training
    - [x] 基于tile的trainer
    - [x] 基于tile的test
      
    - [x] 基于mil的trainer
    - [x] 基于mil的test


### 文件结构

```
.
├── bin
│   ├── core
│   │   ├── dataset.py
│   │   ├── generate_heatmap.py
│   │   ├── mil_models.py
│   │   ├── tester.py
│   │   ├── tile_models.py
│   │   └── trainer.py
│   ├── test.py
│   ├── train_eval.py
│   └── utils
│       ├── config.py
│       ├── data_preparation.py
│       ├── dl_tools.py
│       ├── image_tools.py
│       ├── slide_core.py
│       ├── stain_norm_ref.png
│       └── tools.py
├── data
│   ├── gene_info.csv
│   ├── slides      									# 数字病理图
│   │   ├── slide_0.svs
│   │   └── slide_1.svs
│   ├── thumbnails  									# 缩略图
│   │   ├── slide_0_otsu_image.png
│   │   ├── slide_0_otsu_mask.png
│   │   ├── slide_0_overall.png
│   └── tiles       									# 图像块
│       └── 1_512			   							# {放大倍数}_{图像块大小}
│           ├── data
│           │   ├── slide_0
│           │   └── tile_1.png
│           ├── documents
│           │   ├── all_slides_info.csv					# 所有slide关于tile的信息
│           │   ├── fused_slides_gene_info_mil.csv		# 融合slide和gene信息，用MIL方法，便于后续实验
│           │   ├── fused_slides_gene_info_tile.csv		# 融合slide和gene信息，用tile方法，便于后续实验
│           │   ├── slides_tiles_csv					# 存储所有tile的信息
│           │   │   └── slide_0.csv
│           │   └── train_dataset_tile.csv				# 
│           ├── features								# tile的特征向量，用于后续实验
│           │   └── slide_0.pkl
│           └── tiles_generator.logs					# 生成数据的日志文件
├── results										# 存放结果的文件夹
│   └── trained_models							# 存放训练好的模型
│       └── model_0
│           ├── checkpoints						# 存放训练好的模型权重
│           │   ├── checkpoints_0.pth 
│           │   └── checkpoints_1.pth
│           ├── tb								# 训练过程tensorboard
│           ├── test_result						# 测试结果文件夹
│           │   ├── figures						# 热力图（不一定实现）
│           │   │   ├── slide_0_heatmap.png
│           │   │   └── slide_1_heatmap.png
│           │   └── test_tiles_results			# 关于每一个slide的测试结果
│           │       ├── slide_0.csv
│           │       ├── slide_1.csv
│           │       └── slide_2.csv
│           └── train_model.logs				# 训练过程的日志
│
├── data_preparation.sh							# 预处理数据的shell
├── slide2tile.sh								# 预处理数据的shell
├── test.sh										# 测试的shell
└── train.sh									# 训练的shell
```





