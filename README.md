# Anonymizer Project

I've read several articles on the topic that published individual approaches, but never a combination.

Why not combine the strengths of different systems?

Hashtag#SpaCy can uniquely identify different entities. However, it has weaknesses when it comes to phone numbers.

LLMs can identify different entities.

So, the source text is independently checked by Hashtag#SpaCy and LLM, and a translation list is created. This ensures that identical names and locations in a text are given the same substitution. This helps preserve context.

With this approach, entities are identified independently of one another and the entire set is replaced. 


### Spacy and uv
Normaly you would install the Spacy Model like:

python -m spacy download de_core_news_sm

But if you are using uv this won't work.
You will get the Error Message "No module named pip"

Fortunately, the releases of spaCy models are also released as .whl. So you may install this like:

uv pip install de_core_news_lg@https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl


### Licence and Copyright

This Project is published under AGPLv3 Licence