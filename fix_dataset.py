# URL thetrumparchive.com (FAQ and link is there)
# Direct Download: https://drive.google.com/file/d/16wm-2NTKohhcA26w-kaWfhLIGwl_oX95/view

import json
import re
import sys


if len(sys.argv) > 1:
    try:
        int_in = int(sys.argv[1])
        MAX_TWEETS = int_in
    except:
        MAX_TWEETS = -1 

REGEX_PEOPLE_TAGS   = '(\@[^ ]*)'
REGEX_TWITTER_LINKS = '(http[s]?:\/\/[^ ]*)'

#
# We do not want tags or links. 
def process_text(text, remove_rt = False):
    text = re.sub(REGEX_PEOPLE_TAGS, '', text)
    text = re.sub(REGEX_TWITTER_LINKS, '', text)
    if remove_rt:
        text.replace('RT', '')
    return text


with open('tweets.json', 'rb') as f:
    json_dump = json.load(f)


with open('tweets.txt', 'w+') as f:
    processed = 0
    for i in range(len(json_dump)):
        f.write(process_text(json_dump[i]['text']) + '\n')
        processed = i
        if MAX_TWEETS > 0 and i >= MAX_TWEETS:
            break
            

print(f'Done. Processed {processed} tweets.')
