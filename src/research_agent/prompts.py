PLAN_SYSTEM = """\
You plan web research. Break the user's question into at most {max_items} focused \
sub-questions that together answer it. Each sub-question must work as a web search on \
its own: name the subject, no pronouns, no "it" or "this". Fewer is fine for a simple question.

Reply with JSON only, no prose:
{{"sub_questions": ["...", "..."]}}"""

NOTES_SYSTEM = """\
You take research notes. Read the page excerpt and write up to 5 short bullet points with \
facts that help answer the sub-question. Keep numbers, dates, versions and names exact. \
Use only what the page says.

If the page has nothing useful for the sub-question, reply with exactly: NOTHING"""

NOTES_USER = """\
Sub-question: {sub_question}
Page title: {title}
URL: {url}

Excerpt:
{text}"""

REFLECT_SYSTEM = """\
You check whether research notes are enough to answer a question well. If something \
important is still missing, suggest at most {max_items} follow-up web searches for the gaps. \
Don't repeat searches that were already done.

Reply with JSON only, no prose:
{{"enough": true or false, "missing": "one sentence", "follow_up": ["...", "..."]}}"""

REFLECT_USER = """\
Question: {question}

Searches done so far:
{searched}

Notes:
{notes}"""

WRITE_SYSTEM = """\
You write short research reports in Markdown. Use only the numbered sources you are given. \
Cite every factual claim with its source number in square brackets, like [2] or [1][3]. \
Never invent sources, numbers or URLs.

Do not add a title and do not add a sources list; both are added for you.

Aim for 200 to 400 words: start with a direct answer in two or three sentences, then the \
details as a few short sections or bullet points, then one line on what the sources did \
not settle."""

WRITE_USER = """\
Question: {question}

Sources:
{sources}"""
