import os
from html.parser import HTMLParser
import sys

# the tagesschau Directory where the HTML files are stored
# file_path = "/Users/sifan/Library/Mobile Documents/com~apple~CloudDocs/MA/CL/EET/www.tagesschau.de"

class TextParser(HTMLParser):
    """
    A custom HTML parser to extract the headline, date information, and content
    from an HTML file.
    """
    def __init__(self):
        super().__init__()
        self.headline = []
        self.subheadline = []
        self.content = []
        self.date_info = []
        self.in_span = False
        self.in_headline = False # Flag to indicate if the parser is inside the headline
        self.in_subheadline = False
        self.in_paragraph = False
        # self.in_article_body = False
        self.in_date_info = False
    
    def handle_starttag(self, tag, attrs): # e.g. <tag attr="value">
        if tag == 'span' or tag == 'h3' or tag == 'div': 
            self.in_span = True # Set the flag to ignore content within <span> tags, <h3> tags, and <div> tags
        if tag == 'h1': # for the headline
            self.in_headline = True
        elif tag == 'h2': # for the subheadline
            self.in_subheadline = True
        elif tag == 'p': # for the date information and content
            if any(attr == 'class' and 'metatextline' in value for attr, value in attrs):
                self.in_date_info = True # <p class="metatextline">Stand: 25.06.2024 10:00 Uhr</p>
            else:
                self.in_paragraph = True # for the content
    
    def handle_endtag(self, tag): # the end tag of an element (e.g. </div>)
        if tag == 'span' or tag == 'h3' or tag == 'div':
            self.in_span = False
        elif tag == 'h1':
            self.in_headline = False
        elif tag == 'h2':
            self.in_subheadline = False
        elif tag == 'p':
            self.in_paragraph = False
            self.in_date_info = False
    
    def handle_data(self, data): # process arbitrary data (e.g. text nodes and the content of <script>...</script> and <style>...</style>).
        if self.in_headline:
            self.headline.append(data.strip())
        elif self.in_subheadline:
            self.subheadline.append(data.strip())
        elif self.in_date_info:
            self.date_info.append(data.strip())
        elif self.in_paragraph and not self.in_span:
            self.content.append(data.strip())
    
    def get_text(self): # get all the content of the HTML file
        headline_text = " ".join(self.headline).strip()  
        subheadline_text = " ".join(self.subheadline).strip() 
        date_info_text = " ".join(self.date_info).strip()  
        content_text = " ".join(self.content).strip()
        return headline_text, subheadline_text, date_info_text, content_text

def extract_text_from_html(file_path): # Extract the headline and text from an HTML file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read() 
        parser = TextParser()
        parser.feed(content) # Feed the content of the file to the parser
        return parser.get_text()

def walk_dir_and_print_text(current_dir):
    for dirpath, dirnames, filenames in os.walk(current_dir): # traversing all the files from the current directory
        for filename in filenames:
            if filename.endswith('.html') and filename != 'index.html': # Exclude the index.html file
                file_path = os.path.join(dirpath, filename) # Get the full path of the file
                headline, sub_headline, date_info, content_text = extract_text_from_html(file_path)
                if headline or sub_headline or date_info or content_text: # Check empty or not
                    filename = os.path.splitext(filename)[0] # Remove the file extension '.html'
                    print(f">>>{filename}<<<") # Print the file name
                    print(headline)
                    print(date_info)
                    print(sub_headline)
                    print(content_text)
                    print()

if __name__ == '__main__':
    if len(sys.argv) != 2: # checks whether exactly one argument is passed
        print("Argument error. Please provide the directory path.")
    else:
        input_dir = sys.argv[1] # get the directory path from the command line argument
        walk_dir_and_print_text(input_dir)
