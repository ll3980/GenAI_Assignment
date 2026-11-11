# 使用与提交说明

1. 解压后进入 project 文件夹，应当能看到 Dockerfile、app、models 和 compose.yaml。
2. 打开 Docker Desktop，在 project 中运行 `docker compose up --build`。
3. 浏览器打开 http://localhost:8000/docs 。测试 POST /diffusion/generate，填入 `{"count":2,"seed":123}`；再测试 POST /energy/generate，填入 `{"count":2,"seed":123,"steps":60}`。
4. 直接看图片可打开 http://localhost:8000/diffusion/generate.png?count=16&seed=123 和 http://localhost:8000/energy/generate.png?count=16&seed=123&steps=60 。
5. 先查看 results/assignment4/VALIDATION.md，区分已经完成的检查与尚未完成的 Docker/GitHub 步骤。
6. 将 project 中的文件复制到你本地 GenAI_Assignment 仓库根目录，按 README 的命令 commit 和 push。此环境没有登录你的 GitHub，不能代表你完成远程推送。
7. 英文 PDF 和 THEORY_ANSWERS.md 包含九题的计算过程。完整 ZIP 还包括 API、Docker、训练代码、模型权重和 Colab notebook。

包内是实际在 CIFAR-10 上训练的小型模型，用于演示训练、梯度控制和 API 流程。生成图仍可能有明显噪声。需要提高画质时，将 notebooks/Assignment4_Training.ipynb 上传到 Colab，选择 GPU，运行更长的训练。下载新权重替换 models 中的同名文件，再重建 Docker。Notebook 中未执行的长训练不能写成已经完成的实验。

初次 Docker 构建会下载依赖和原有 spaCy 词向量模型，需要联网。推理本身使用随包权重，不下载 CIFAR-10。统一入口是根目录的 app/main.py，main.py 是同一应用的别名。
