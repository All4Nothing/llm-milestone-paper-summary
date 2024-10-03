# LLM Milestone Papers Summary

### Attention Is All You Need

- Date: 2017.06
- Keyword: Transformers
- Institute: Google

![Transformer](https://github.com/user-attachments/assets/e4bec0ef-07c4-4067-b925-d3f1de439726)

> Attention Is All You Need는 2017년 Google의 Research팀이 발표한 논문으로 Transformer Architecture를 처음으로 소개한다. 이 논문에서는 RNN이나 CNN을 사용하지 않고, 오직 Attention Mechanism만을 사용하는 새로운 simple network architecture인 Transformer를 제안한다. Transforemr는 병렬 처리가 가능하며, 더 적은 학습시간으로 우수한 성능을 달성했다.

> 주요한 특징으로는 Transformer는 Encoder-Decoder 구조이며, Self-Attetnion 레이어, Positional Encoding, Multi-Head Attention을 사용한다. 긴 sequence의 의존성을 효과적으로 학습하고, 병렬 처리로 인한 학습 시간을 단축할 수 있었으며, 해석 가능성이 높다는 장점이 있다.

> 이 논문은 자연어 처리 분야에 혁명적인 변화를 가져왔으며, 현재 거의 모든 LLM(Large Language Model)의 기반이 되었다.

### Improving Language Understanding by Generative Pre-Training

- Date: 2018.06
- Keyword: GPT 1.0
- Institude: OpenAI

![GPT-1](https://github.com/user-attachments/assets/b7e518ec-4c14-49f4-b386-154f5a93c746)


> Improving Language Understanding by Generative Pre-Training는 OpenAI의 GPT-1을 소개한다. 이 논문에서는 unlabeled된 대규모 텍스트 데이터로 사전학습(pre-training)을 수행한 후, 특정 task에 대해 미세조정(fine-tuning)을 하여 모델을 학습시키는 방식을 제안한다. 이 논문에서는 Transformer decoder architecture를 기반으로 한 모델인 GPT-1을 소개한다.

> 주요한 특징으로는 GPT-1은 unlabeled된 대규모 텍스트 데이터를 바탕으로, 다음 단어 예측를 예측하는 unsupervised pre-training을 수행하고, 특정 task에 맞게 모델을 fine-tuning하는 supervised learning 기반의 fine-tuning을 진행한다.

> 이 논문은 LLM 모델의 시대를 열었으며, GPT-2, GPT-3 등 후속 연구의 기반이 되었다.

### BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

- Date: 2018.10
- Keyword: BERT
- Institude: Google

![BERT](https://github.com/user-attachments/assets/249fc5ec-59b4-46ba-a8b1-6dce26e3cfd7)


> BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding는 Bidirectional Encoder Representations from Transformers, BERT를 소개한다. BERT는 양방향(bidirectional) Transformer를 사용하여 unlabled 텍스트로부터 deep bidiretional representation을 학습한다. BERT는 Masked Language Model(MLM)과 Next Sentence Prediction(NSP) task로 사전 학습(pre-training)을 수행한다.

> 주요한 특징으로는 BERT는 기존 모델들과 달리 모든 layer에서 좌우 문맥(context)를 동시에 고려하는 양방향(bidirectional) architecture이다. BERT는 MLM과 NSP라는 새로운 방식으로 사전 학습을 수행한다. Masked Language Model, MLM은 입력의 일부를 무작위로 마스킹(Masking)하고, 마스킹된 단어를 예측하는 기법이다. Next Sentence Prediction, NSP는 두 문장이 실제로 연속된 문장인지를 예측하는 task를 수행하여 문서 간의 관계를 학습하는 기법이다.

> 이 논문은 자연어 처리 분야에 새로운 기준을 제시했으며, RoBERTa, ALBERT 등 후속 연구의 기반이 되었다.

### Language Models are Unsupervised Multitask Learners

- Date: 2019.02
- Keyword: GPT 2.0
- Institude: OpenAI

> Language Models are Unsupervised Multitask Learners는 GPT-2를 소개한다. 이 논문에서는 supervision없이도 다양한 NLP task를 수행할 수 있는 모델인 GPT-2를 소개하였다. GPT-2는 task 특정 데이터셋이나 architecture의 수정 없이 zero-shot task를 수행하며, 더 큰 모델과 더 다양한 데이터로 성능 향상을 이뤘다.

> 주요한 특징으로는 이 논문에서는 높은 품질의 웹 문서 데이터셋은 WebText 데이터셋을 구축하였고, GPT-1 architecture를 기반으로 확장한 모델을 사용했다.

> 이 논문은 언어 모델이 특별한 supervise d learning없이도 다양한 NLP task를 수행할 수 있다는 것을 보였으며, task를 자연어로 설명하는 것만으로도 해당 task를 수행할 수 있는 zero-shot learning의 가능성을 보였다.

### Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

- Date: 2019.09
- Keyword: Megatron-LM
- Institude: NVIDIA

![Megatron-LM](https://github.com/user-attachments/assets/66a2cf79-32c9-498c-ad05-05a537c178dd)


> Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism은 수십억개의 parameters를 가진 Large Language Model을 학습하는 효율적인 방법을 제시한다. 이 논문에서는 큰 모델을 여러 GPU에 나눠 배치하여 학습하는 방법인 Model Prallelism(모델 병렬화)를 통한 분산 학습을 구현한다.

> 주요한 특징으로는, 모델 병렬화를 구현하기 위해 Transformer의 self-attention과 MLP layer를 각각 병렬화하였고, 텐서 병렬화(Tensor Parallelism)을 통한 효율적인 분산 처리를 수행한다.

> 이 논문은 LLM의 효율적인 학습을 위한 기술적인 혁신을 제시했으며, 특히 모델 병렬화를 통한 분산 학습 방법론을 확립하였다.

### Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

- Date: 2019.10
- Keyword: T5
- Institude: Google

![T5](https://github.com/user-attachments/assets/d7251db6-7b35-43a7-b34f-5efa82efb1e3)


> Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer는 모든 NLP 문제를 Text-to-Text 형식으로 전환하여 일관된 방식으로 처리하는 transformer 모델, T5를 소개한다.

> 주요한 특징으로는 Text-to-Text Transformer, T5는 입력과 출력을 모두 텍스트 형식으로 통일한다. 예를 들어 translation(번역) 문제는 *"translate English to German: The house is wonderful." → "Das Haus ist wunderbar.”*와 같이 전환하고, 요약 문제는 "summarize: long article" → "short summary”와 같은 방식으로 전환한다. 또한, 웹 크롤링 데이터를 cleaning(정제)한 Colossal Clean Crawled Corpus, C4 데이터셋을 사용한다. 

> 이 논문은 NLP task를 Text-to-Text 형식으로 접근하는 새로운 패러다임을 제시했다.

### ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

- Date: 2019.10
- Keyword: ZeRO
- Institude: Microsoft

> ZeRO: Memory Optimizations Toward Training Trillion Parameter Models는 기존 모델 병렬화와 파이프라인 병렬화의 장점을 결합하여, 데이터 병렬화(Data Parallelism)의 메모리 효율성을 획기적으로 개선하는 방법인 Zero Redundancy Optimizer, ZeRO를 소개한다.

> 주요한 특징으로는, ZeRO는 세가지 단계로 메모리 중복(redundancy)를 제거한다. 먼저, ZeRO-1은 Adam optimizer의 모멘텀 및 분산 상태를 분할하고, ZeRO-2는 gradient를 데이터 병렬 프로세스 간에 분할하며, ZeRO-3은 모델 parameters를 분할한다. 이를 통해 기존 방법 대비 최대 40배 메모리 효율을 개선하였다.

> 이 논문은 대규모 AI 모델 학습의 핵심 병목(bottleneck)문제인 메모리 문제를 혁신적으로 해결하였다.

### Scaling Laws for Neural Language Models

- Date: 2020.01
- Keyword: Scaling Law
- Institude: OpenAI

![Scaling Laws](https://github.com/user-attachments/assets/de3c8517-70ae-4279-a9ef-dc830353816c)


> Scaling Laws for Neural Language Models는 언어 모델의 성능과 parameters 수, 데이터셋의 크기, 계산량 간의 관계를 체계적으로 분석하였다.

> 주요한 특징으로는, 모델 성능은 N(모델 parameters 수), D(데이터셋의 크기), C(계산량)과 멱법칙(Power Laws) 관계를 보인다( $L ∝ N^{-0.076}$, $L ∝ D^{-0.095}$, $L ∝ C^{-0.050}$이며, 여기서 $L$은 test loss). 또한, N, C, D 간의 trade-off가 존재하는데, 계산량(C)이 4배 증가할 때 최적의 모델 크기(N)는 2배로 증가하며, 모델 크기(N)가 증가할 수록 필요한 데이터셋의 크기(D)도 증가하고, 작은 모델을 오래 학습하는 것보다 큰 모델을 짧게 학습하는 것이 효과적이다.

> 이 논문은 Language 모델 스케일링(Scaling)에 대한 정량적 기반을 제공하였고, 특히 모델 크기, 데이터셋 크기, 계산량 간의 관계를 명확한 수학적 형태로 제시했다는 점이 혁신적이었다.

### Language Models are Few-Shot Learners

- Date: 2020.05
- Keyword: GPT 3.0
- Institude: OpenAI
  
![GPT-3](https://github.com/user-attachments/assets/b5bd6e26-00b2-49af-8987-86f39e33c9ec)


> Language Models are Few-Shot Learners는 OpenAI에서 개발한 1,750억개의 parameters를 가진LLM 모델, GPT-3를 소개한다. 

> 주요한 특징으로는, 모델의 크기를 크게 늘림으로써 few-shot learning 성능이 지속적으로 향상되며, 별도의 미세조정 없이도 다양한 NLP task를 수행할 수 있다. Few-shot learning이란 task별 fine-tuning없이도 몇 가지 예시를 제시함으로써 새로운 task를 수행 가능하다는 것이다.

> 이 논문은 모델 규모의 확장이 few-shot learning 성능 향상으로 이어질 수 있음을 입증하였으며, few-shot learing 능력을 보여주는 모델을 소개하였다는 점이 획기적이었다.

### Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

- Date: 2021.01
- Keyword: Switch Transformers
- Institude: Google

![Switch Transformers](https://github.com/user-attachments/assets/7dd622b6-1026-4081-b4bb-8f56dcb53663)


> Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity은 LLM을 효율적으로 학습할 수 있는 단순하고 효과적인 MoE routing 알고리즘을 소개한다.

> 주요한 특징으로는, 기존 Transformer의 FFN layer를 Mixture of Experts(MOE) layer로 대체하여, 입력toknen에 따라 동적으로 expert를 선택하고, 각 token별 적합한 expert를 뽑는 routing 알고리즘 방식을 바꿔 routing 구현을 간단하게 하면서 계산 비용을 줄였다.

> 이 논문은 모델의 parameters 수를 크게 늘리면서도 계산 효율성을 향상시켰다는 점에서 혁신적이었다.
