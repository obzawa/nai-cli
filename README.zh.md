# nai-cli

[English](README.md) | 中文

nai-cli 是 [NovelAI](https://novelai.net) 图像生成的非官方命令行客户端，支持 V5、V4.5 与 V4 的文生图、图生图、局部重绘、Enhance、放大、导演工具、Vibe 与精准参考，并能读出图片中保存的参数以重新生成。

它按官网前端的规则组装请求，相同的参数与种子会生成与官网相同的图像，例外见[规则对照](docs/official-parity.md#intentional-differences)。每条命令输出 JSON（一个对象，或每个任务一行），便于在脚本和其他程序中调用。

文档（英文）：[使用说明](docs/usage.md)、[官网规则对照](docs/official-parity.md)、[兼容性](docs/usage.md#compatibility)

## 安装

需要 Python 3.11 或更高版本：

```sh
pipx install nai-cli
```

该命令会把 `nai` 安装在独立的环境中，也可以改用 `uv tool install nai-cli` 或 `pip install nai-cli`。反推标签和按物体抠蒙版在本机运行，需要安装 `pipx install "nai-cli[local]"`。

## 配置

在 NovelAI 的账号设置中生成持久 API 令牌（以 `pst-` 开头），然后创建配置文件：

```sh
nai config init
```

该命令会输出新配置文件的路径。将令牌填入其中的 `novelai_token`，或改为设置环境变量 `NOVELAI_TOKEN`。令牌生效后，`nai account` 会显示账户余额。

## 使用

```sh
nai gen --prompt "1girl, rooftop, night" --dry-run
nai gen --prompt @prompt.txt --size landscape
nai inpaint --image a.png --region ellipse:0.62,0.40,0.80,0.58 --prompt "1girl, rooftop, night, open hand" --preview
nai enhance --image a.png
nai meta a.png
nai batch jobs.jsonl --dry-run
```

`--dry-run` 只组装请求并估价，不发送付费请求。输出中的 `prompt` 与 `negative` 是最终发送的文本，包含自动添加的质量词和 UC 预设；去掉 `--dry-run` 即正式生成，保存的文件列在 `images` 中。`@prompt.txt` 表示从文件读取提示词。`inpaint` 按图像比例坐标重绘指定区域并贴回原图，`--preview` 会在发送前保存一张标出该区域的预览图。`enhance` 按图像中保存的参数放大精修，`meta` 读取这些参数，供 `nai gen` 复现该图像。`batch` 按行执行一组命令，每条输出一行 JSON，便于其他程序排队提交，失败后接着跑剩下的。

`nai --help` 列出全部命令，`nai <命令> --help` 显示各命令的参数。各命令、输出格式与错误码详见[使用说明](docs/usage.md)。

## Agent 技能

nai-cli 附带一个 [Agent Skills](https://agentskills.io) 格式的技能，指导 AI agent 使用本工具：消耗 Anlas 前先试运行、超出单次花费上限前先征得同意、处理错误，以及局部重绘的操作流程。安装方式：

```sh
npx skills add obzawa/nai-cli
```

默认安装到当前项目，加 `-g` 则对所有项目生效。技能文件位于 [skills/nai-cli](skills/nai-cli)。

## 开发

环境搭建、提交改动前要通过的检查和报告问题的方式见 [CONTRIBUTING.md](CONTRIBUTING.md)；编程 agent 遵循 [AGENTS.md](AGENTS.md)。

## 致谢

感谢 [NovelAI](https://novelai.net) 提供模型与服务。许多默认参数与便利功能参考了 [NAI Launcher](https://github.com/Aaalice233/Aaalice_NAI_Launcher)，图像缩放移植自 [Pica](https://github.com/nodeca/pica)。

## 许可证

[MIT](LICENSE)

第三方代码及其许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
