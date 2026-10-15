# 运行与提交

本包已基于之前的 Assignment 1 API 合并 CNN，保留 `/generate` 和 `/embedding`。CNN 已训练完成，权重在 `models/cifar10_cnn.pt`，不需要先下载 CIFAR-10 或重新训练。

## 1. 解压并运行 Docker

启动 Docker Desktop。在解压后的项目目录打开终端，运行：

```bash
docker compose up --build
```

第一次构建会安装依赖和旧 embedding 接口使用的 spaCy 预训练模型。看到服务器已启动后，打开：

http://127.0.0.1:8000/docs

找到 **POST /predict**，点击 **Try it out**，选择 `examples` 中任意一张图片，再点击 **Execute**。成功时返回 HTTP 200，并给出 `class_name`、`class_id`、`confidence` 和十类概率。

也可在另一个终端检查全部接口：

```bash
python scripts/check_api.py --base-url http://127.0.0.1:8000 --output results/docker_api_check.json
```

这个检查脚本只需要 Python 标准库。它通过实际 HTTP 请求检查 CNN、文本生成和 embedding，全部成功才会写出结果文件。

## 2. 放进原 GitHub 项目

将本包的项目文件放入你原来的 GitHub 项目目录，保留原仓库的 `.git` 目录。`pyproject.toml` 和 `uv.lock` 要一起更新。`app/main.py` 已加入 CNN router，`Dockerfile` 已包含复制训练权重的步骤。

如果只想添加 CNN 的几个文件，按照 `INTEGRATION.md` 操作。

在原仓库检查变更，然后提交并推送。需提交的内容包括代码、依赖文件、Dockerfile、训练权重、训练结果和理论答案；CIFAR-10 原始数据不需要提交。不要仅把 ZIP 上传到 GitHub 后当作代码提交，仓库中应能直接看到项目文件。

## 3. CourseWorks 提交

提交 `Assignment2_Theory_LiuyangLi.pdf`，并在提交说明中附上实际 GitHub 仓库链接。需要把代码一并作为附件时，可以上传整个项目 ZIP；ZIP 内已包含理论 PDF。

评分中有 GitHub 提交和 Docker/API 部署，所以仅上传理论 PDF 不能覆盖所有评分项。

## 已完成的模型

- 架构：严格按附件，输入为 64 x 64 RGB，Flatten 为 8192。
- 数据：45,000 张训练、5,000 张验证、10,000 张测试。
- 训练：10 轮，按验证准确率选中第 8 轮。
- 验证准确率：69.48%。
- 最终测试准确率：68.50%（6,850 / 10,000）。
- 理论：六题英文答案，均附推导或解释。

具体已验证内容及环境限制见 `results/VERIFICATION.md`。远程 GitHub 提交需要在你自己的仓库完成；本次没有修改远程仓库。
