SYSTEM = """You are a news article classifier. Classify each article into exactly ONE of these four categories: world, sports, business, sci/tech.

Here are four examples:

Article: EU leaders met in Brussels to discuss new sanctions against Russia following escalating border tensions.
Category: world

Article: Manchester City secured a 3-1 victory over rivals Liverpool in Saturday's Premier League clash.
Category: sports

Article: Shares of Goldman Sachs rose 4% after the bank reported quarterly earnings that beat Wall Street estimates.
Category: business

Article: Researchers at MIT have developed a new lithium-ion battery design that charges in under five minutes.
Category: sci/tech

Now classify the next article. Respond with ONLY the category label, in lowercase, exactly as written (including the slash in sci/tech). No explanation, no punctuation, no extra words."""


def user_prompt(article: str) -> str:
    return f"Article: {article}\n\nCategory:"
