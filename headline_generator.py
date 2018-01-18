import feedparser
from urlparse import urlparse


class RSSParser():
    
    def __init__(self):
        self.f = feedparser.parse('https://news.google.com/news/rss/headlines/section/topic/BUSINESS?ned=us&hl=en')
        self.top_articles = self.f.entries
        self.article_dicts = []
        example_article = top_articles[0]
        example_article_keys = example_article.keys()

        for key in example_article_keys:
            print str(key)+': '+str(example_article[key])

    def get_headline_packets(self):
        for article in self.top_articles:
            _tmp = {}
            _tmp['published'] = article['published']
            _tmp['headline'] = article['title_detail']['value']
            parsed_uri = urlparse(article['link'])
            domain = '{uri.netloc}'.format(uri=parsed_uri)
            print domain
            _tmp['org'] = domain
            self.article_dicts.append(_tmp)

        self.article_dicts[0]