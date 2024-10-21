import os
from html.parser import HTMLParser
import sys

class ArticleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.layout_depth = 0
        self.is_title = False
        self.is_content = False
        self.is_within_layout = False
        self.title = []
        self.content = []


    def handle_starttag(self, tag, attrs):
        if tag == 'div' and ('class', 'layout-container') in attrs:
            self.is_within_layout = True

        if self.is_within_layout:
            # print("in layout")
            if tag == 'h1':
                self.is_title = True
            elif tag in ('h2', 'h3', 'p'):
                self.is_content = True

    def handle_endtag(self, tag):
        # if tag == 'div' and self.is_within_layout:
        #     print("end layout")
        #     self.is_within_layout = False
        #     self.is_layout_div = False
        if tag == 'h1':
            self.is_title = False
        if tag in ('h2', 'h3', 'p'):
            self.is_content = False

    def handle_data(self, data):
        if self.is_title:
            cleaned_title = data.strip()
            self.title.append(cleaned_title)
            # print(self.title)
        elif self.is_content:
            cleaned_data = data.strip().replace('\n', ' ').replace('\r', '')
            self.content.append(cleaned_data)

    def get_data(self):
        format_title = ' '.join(self.title).replace('  ', ' ').strip()
        combined_content = ' '.join(self.content)
        format_content = combined_content.replace('  ', ' ').strip()
        return format_title, format_content


def extract_articles(directory):
    for root, dir, files in os.walk(directory):
        for filename in files:
            if filename == 'index.html':
                continue
            file_path = os.path.join(root, filename)
            # try:
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
                parser = ArticleParser()
                parser.feed(html_content)
                title, content = parser.get_data()
                # print(title)
                print(f"¿¿¿{filename}¡¡¡")
                print(title)
                print(content)


            # except Exception as e:
            #     print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    directory = sys.argv[1]
    extract_articles(directory)







