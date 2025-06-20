## 任务介绍

本次评测任务的文本语料包含大模型生成文本和真实人类文本两部分。其中，人类文本来源于互联网上真实人类的评论、写作、新闻等内容，而大模型生成文本包含来源于7个主流大模型生成的文本，所有数据按照10:1的比例随机均匀划分训练集和测试集。任务目标是给定输入文本，正确分类其为大模型生成文本（标签为1）还是人类撰写文本（标签为0）。

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