# <img src=assets/wildteaming_logo.png width=40/> WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models

<p align="center">

<a href="https://arxiv.org/abs/2406.18510"><img src="https://img.shields.io/badge/📝-paper-blue"></a>
<a href="https://huggingface.co/datasets/allenai/wildjailbreak"><img src="https://img.shields.io/badge/🤗-wildjailbreak (data)-orange"></a>
<a href="https://huggingface.co/allenai/llama2-7b-WildJailbreak"><img src="https://img.shields.io/badge/🤗-tulu2--7b--wildjailbreak (model)-green"></a>
<a href="https://huggingface.co/allenai/llama2-13b-WildJailbreak"><img src="https://img.shields.io/badge/🤗-tulu2--13b--wildjailbreak (model)-green"></a>
<a href="https://github.com/allenai/wildteaming"><img src="https://img.shields.io/badge/🔗-code-red"></a>

</p>

**Authors:**
[Liwei Jiang](https://liweijiang.me),
[Kavel Rao](https://kavelrao.dev) ⭐,
[Seungju Han](https://seungjuhan.me) ⭐,
[Allyson Ettinger](https://aetting.github.io),
[Faeze Brahman](https://fabrahman.github.io),
[Sachin Kumar](https://sites.google.com/view/sachinkumar),
[Niloofar Mireshghallah](https://homes.cs.washington.edu/~niloofar/),
[Ximing Lu](https://scholar.google.com/citations?user=ssYPSmkAAAAJ&hl=en),
[Maarten Sap](http://maartensap.com),
[Yejin Choi](https://homes.cs.washington.edu/~yejin/),
[Nouha Dziri](https://nouhadziri.github.io/)
&nbsp; &nbsp; &nbsp; ⭐ Co-second authors

We introduce <img src=assets/wildteaming_logo.png width=25/> [WildTeaming](https://arxiv.org/pdf/2406.18510), an automatic red-teaming framework that mines *in-the-wild* user-chatbot interactions to discover 5.7K unique clusters of novel jailbreak tactics, and then composes selections of multiple mined tactics for systematic exploration of novel and even more challenging jailbreaks. WildTeaming intends to address two challenges: 
- 🔍 Broadly identifying jailbroken behaviors of LLMs.
- 🛠️ Creating a publicly open, large-scale safety training resource for systematic defense (WildJailbreak).

For more findings, please refer to our [paper](https://arxiv.org/abs/2406.18510)!

