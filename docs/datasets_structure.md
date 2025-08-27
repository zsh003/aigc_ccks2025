## 大赛概况介绍

随着人工智能技术的不断发展，人工智能生成内容（AIGC）的质量日益提升，尤其是大型语言模型生成的文本，已经逐步接近甚至在某些场景中超越人类撰写的水平。因此，AIGC在教育、媒体、商业以及社交平台等领域得到了广泛的应用。然而，大模型技术的高速发展也带来了诸多挑战，例如学术不端、虚假信息传播以及有害内容的生成等负面问题。面对这些问题，如何准确地区分由大模型生成的文本与人类撰写的文本，成为当前学术界和工业界亟需解决的重要课题，并对于维护信息生态安全、促进技术规范发展以及国家层面的监管具有深远意义。

大模型生成文本检测的挑战主要体现在以下几个方面：（1）文本风格日益逼真：随着模型规模的扩大和训练数据的丰富，生成文本在语法结构、上下文连贯性以及语义表达方面日益接近人类写作风格，传统基于语言特征的检测方法逐渐失效。（2）模型与训练数据的黑盒性：多数主流大模型为闭源产品，外界难以获取其训练数据和参数细节，进而难以构建针对性的检测机制。此外，同一模型在不同参数设置或提示语下生成的文本差异较大，进一步增加了检测难度。（3）对抗性伪装与改写：恶意用户可通过同义改写、语序调整或语言风格变换等方式对生成文本进行伪装，以规避检测系统，从而增强生成内容的“人类伪装性”。（4）数据分布漂移与新模型涌现：随着新一代语言模型不断推出，其生成风格可能与现有检测方法的训练样本存在分布差异，导致检测准确率下降。因此，尽管这项任务具有重要的应用价值，但要实现高效准确的检测机制，仍需克服上述挑战。

中国科学院信息工程研究所在CCKS2025大会组织本次评测任务。本次评测将依托阿里云天池平台展开。

## 任务介绍

本次评测任务的文本语料包含大模型生成文本和真实人类文本两部分。其中，人类文本来源于互联网上真实人类的评论、写作、新闻等内容，而大模型生成文本包含来源于7个主流大模型生成的文本，所有数据按照10:1的比例随机均匀划分训练集和测试集。任务目标是给定输入文本，正确分类其为大模型生成文本（标签为1）还是人类撰写文本（标签为0）。

## 参赛规则

1.所有参赛选手都必须在天池平台中注册、报名，本次比赛的参赛对象仅限全日制在校大学生（本科、硕士、博士均可）和企业员工；
2.参赛选手需确保注册时提交信息准确有效，所有的比赛资格及奖金支付均以提交信息为准；
3.参赛选手在天池中组队，参赛队伍成员数量不得超过5个，报名截止日期之后不允许更改队员名单；
4.每支队伍需指定一名队长，队伍名称不超过15个字符，队伍名的设定不得违反中国法律法规或公序良俗词汇，否则组织者有可能会解散队伍；
5.每名选手只能参加一支队伍，一旦发现某选手以注册多个账号的方式参加多支队伍，将取消相关队伍的参赛资格；
6.允许使用开源代码或工具，但不允许使用任何未公开发布或需要授权的代码或工具；
7.参赛选手允许基于训练数据进行数据增强，例如通过裁剪、拆分、词语替换或格式调整等方法生成新的数据样本，但须确保严格保留原始数据的语义，不得引入任何外部知识或生成完全新创的内容，不能用额外的标注数据。
8.禁止使用生成式大模型进行释义(paraphrase)操作，允许使用传统的编码器模型或seq2seq模型进行释义操作，前提是严格遵守语义保留的要求。
9.参赛队伍可在参赛期间随时上传验证集的预测结果，一天不能超过3次 ，排行榜每小时整点更新。

## 数据集介绍

本次比赛的数据主要来自通用领域的公开新闻、报道等，所有数据将以JSONL对象的格式进行存储：

数据集路径：../datasets

### 训练集

训练集涵盖7种大模型：GPT-4o, DeepSeek, Llama3, ChatGPT, GLM-4, Qwen2.5, Claude-3，数据来源涵盖ELI5（问答）、BBC News（新闻写作）、ROC stories（故事生成）、Abstracts（学术写作）、IMDB（评论表达）、Wikipedia（知识解释）共6种任务，训练数据总共包含28000条样本，人类和大模型文本比例为1:1。具体而言，其数据示例如下所示：

```json
{"text": "Registering a Limited Liability Company (LLC) in a foreign state—meaning a state other than the one where you primarily conduct business—can be a strategic decision, but it involves certain considerations and potential issues:\n\n1. **Foreign Qualification**: If you form an LLC in one state but do business in another, you'll need to register as a foreign LLC in the state where you conduct business. This involves filing additional paperwork and paying fees.\n\n2. **Asset Protection and Liability**: Some states offer stronger asset protection laws than others, which might influence your choice. However, operating in multiple states could complicate liability issues.\n\nBefore deciding to register an LLC in a foreign state, it’s advisable to consult with legal and tax professionals who can provide guidance based on your specific business needs and goals.", "label": 1}
{"text": "Basically there are many categories of \" Best Seller \" . Replace \" Best Seller \" by something like \" Oscars \" and every \" best seller \" book is basically an \" oscar - winning \" book . May not have won the \" Best film \" , but even if you won the best director or best script , you 're still an \" oscar - winning \" film . Same thing for best sellers . Also , IIRC the rankings change every week or something like that . Some you might not be best seller one week , but you may be the next week . I guess even if you do n't stay there for long , you still achieved the status . Hence , # 1 best seller .", "label": 0}
```

### 测试集

测试集分A榜测试集和B榜测试集，分别包含2800条数据，未知来源，只包含文本内容，不包含标签，其数据样式如下：

```json
{"text": "Okay! Imagine your mind is like a TV with lots of channels, and each channel is a different thought or feeling. Sometimes, it’s hard to change the channel because there are so many things going on at once.\n\nHypnotism is like having a magical remote control that helps you focus on just one channel at a time. A hypnotist helps you relax and concentrate so you can listen to just one thought. It’s like when someone reads you a bedtime story and you get all cozy and calm.\n\nIn this calm state, you might imagine fun things, like being on a beach or floating on a cloud. It helps you feel relaxed and can sometimes make it easier to learn new things, feel better, or even stop bad habits, like biting your nails.\n\nRemember, it’s all about being super relaxed and using your imagination!" }
```