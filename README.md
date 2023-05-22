# SPT-ABSA

本仓库开源了以下论文的代码：

- 标题：An Empirical Study of Sentiment-Enhanced Pre-Training for Aspect-Based Sentiment Analysis
- 作者：Yice Zhang, Yifan Yang, Bin Liang, Shiwei Chen, Bing Qin, and Ruifeng Xu
- 会议：ACL-2023 Finding (Long)

模型权重可以在 [https://huggingface.co/zhang-yice/spt-absa-bert-400k](https://huggingface.co/zhang-yice/spt-absa-bert-400k) 获取。

### What Did We Do?

Aspect-Based Sentiment Analysis (ABSA) is an important problem in sentiment analysis.
Its goal is to recognize opinions and sentiments towards specific aspects from user-generated content.
Many research efforts leverage pre-training techniques to learn sentiment-aware representations and achieve significant gains in various ABSA tasks.
We conduct an empirical study of SPT-ABSA to systematically investigate and analyze the effectiveness of the existing approaches.

We mainly concentrate on the following questions: 
- (a) what impact do different types of sentiment knowledge have on downstream ABSA tasks?;
- (b) which knowledge integration method is most effective?; and
- (c) does injecting non-sentiment-specific linguistic knowledge (e.g., part-of-speech tags and syntactic relations) into pre-training have positive impacts?

Based on the experimental investigation of these questions, we eventually obtain a powerful sentiment-enhanced pre-trained model.
The powerful sentiment-enhanced pre-trained model is zhang-yice/spt-absa-bert-400k, which integrates three types of knowledge:
- aspect words: masking aspects' context and predicting them.
- review's rating score: rating prediction.
- syntax knowledge: 
  - part-of-speech,
  - dependency direction,
  - dependency distance.
 
### Experimental Results

<img width="75%" alt="image" src="https://github.com/HITSZ-HLT/SPT-ABSA/assets/9134454/38fc2db0-6ccf-47a7-a93c-cf54667e1a23">

<img width="75%" alt="image" src="https://github.com/HITSZ-HLT/SPT-ABSA/assets/9134454/20c5a976-014e-433f-a2ec-4bb259e5a382">

