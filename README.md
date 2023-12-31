# Sales bot for electronic device sales

### Features
* A QA generation chain based on customized prompts to generate QA pairs based on scenarios

* Subclassing LLM for Api2d (cheaper and avoid great wall blocking on openai)

* Changed embedding to local model cached from huggingface for better network and Chinese embedding support.

* Customized sales ZeroShotAgent to search vectordb firs, then web, finally llm for the answer.

* [TODO] Database query function to retrieve product spec or pricing.

* [TODO] Fewshot on sales talk techniques on bargaining with customer. See [blog](https://zhuanlan.zhihu.com/p/357487465) about sales logic.
  
* [TODO] Multi modal Q&A accepting apartment structure images and 
  return mocked decoration effect images.

![Alt text](resources/image.png)