'''from tokenize import PlainToken
from torchvision import datasets
import numbers
import torch
import matplotlib.pyplot as plt
import os
from bs4 import BeautifulSoup as Soup
import requests
import sys
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from transformers import pipeline
#import numpy as np
from urllib.request import Request
from  pyarabic.araby import strip_harakat'''
'''import torchaudio
from torchvision.io.image import read_image
#from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image'''
'''
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

text = " hi how are you i am younis"
with torch.inference_mode():
    processed, lengths = processor(text)
    processed = processed.to(device)
    lengths = lengths.to(device)
    spec, spec_lengths, _ = tacotron2.infer(processed, lengths)
    waveforms, lengths = vocoder(spec, spec_lengths)
torchaudio.save("output_wavernn.wav", waveforms[0:1].cpu(), sample_rate=vocoder.sample_rate)'''

'''
def exifsite(query):
  url="https://api.binaryedge.io/v2/query/search?query={}"
  data={"X-Key":"30e21f4b-b6ca-4964-a356-9cd1e83e7b83"}
  a=requests.get(url=url.format(query),headers=data)
  print(a.json())

def find_file_from_s3_aws(keyword):
 data={"keywords": keyword ,
 "order": "unordered",
 "extensions":"" }
 list=[]
 numberpage=[]
 respone=False
 try:
  b=requests.get(url="https://buckets.grayhatwarfare.com/results/{}".format(keyword))
  soup = BeautifulSoup(b.text, "html.parser")
  respone=True
  for tag in soup.select("td:has(a)"):
   link=tag.find("a")["href"] 
   if link.endswith(".com"):
    continue
   else:
     list.append(link)
 except:
    print("error in request please try again later")
 if respone:
  b=requests.get(url="https://buckets.grayhatwarfare.com/results/{}/{}".format(keyword,2))
  soup = BeautifulSoup(b.text, "html.parser")
  for tag in soup.select("td:has(a)"):
   link=tag.find("a")["href"] 
   if link.endswith(".com"):
    continue
   else:
     list.append(link)
 print(list)
 return list

exifsite(query="198.168.0.1")'''

'''for tag in soup.select("li.page-item"):
   s__tring=tag.stringcountry:BH&port:80
   numberpage.append(s__tring)'''
'''
url = "https://jokeapi-v2.p.rapidapi.com/joke/Programming"
querystring = {"idRange":"0-150","blacklistFlags":"nsfw,racist"}
headers = {
"X-RapidAPI-Key": "764905dd0cmshc891402bd00dd69p1570b7jsn7abdba66fb03",
	 "X-RapidAPI-Host": "jokeapi-v2.p.rapidapi.com"
        }
response = requests.get(url, headers=headers, params=querystring)
print(response.text)
response=response.json()
response=response["joke"]
print(response)
a=requests.get(url="https://thor-graphql.dictionary.com/v2/search?searchText=agument&context=dictionary")
b=a.json()
if b["count"] != 0:
 print(len(b['data'][0]))
 for i in range(0,len(b['data'])):
  print(b['data'][i]["reference"]["source"]["redirectUrl"])
elif b["count"] == 0:
  print("a")'''

country="ن"
'''
tran=requests.get(url="https://timesprayer.today/ajax.php?do=localsearch&keyword={}".format(country))
print(tran.text)

print(len(tran.json()))
a=str(u'ci65-\u0645\u0648\u0627\u0642\u064a\u062a-\u0627\u0644\u0635\u0644\u0627\u0629-\u0641\u064a-\u0627\u0644\u062d\u062f\u064a\u062f\u0629')#.format(tran.json()[0]["alies"]
print(a)
b=strip_harakat(a)
#tran.content.decode("windows-1256"))

#print(a.text)
from youtube_search import YoutubeSearch

results = YoutubeSearch('برد الشتاء مايحن زود على المشتحن', max_results=10).to_dict()

print(results)
for i in range(0,len(results)):
 print(results[0]["url_suffix"])'''
'''
times=[]
tran=requests.get(url="https://timesprayer.com/prayer-times-cities-bahrain.html")
img = Image.open('car.png')
I1 = ImageDraw.Draw(img)
myFont = ImageFont.truetype('FreeMono.ttf', 65)
I1.text((10, 10), "Nice Car", font=myFont, fill =(255, 0, 0))
img.save("car2.png")
soup = Soup(tran.text)
div = soup.find_all("li", {"class": "onprayer"})
for i in div:
   a= i.find("time").text
   b=i.find("div").text
   times.append(b +":"+a)
print(times)'''
import requests
import json
#"uuid_idempotency_token": "76458788-146e-4de6-bd90-6870193266fa"'''
'''header={
'authority': 'api.fakeyou.com',
'method':'POST',
'path': '/tts/inference',
'scheme': 'https',
'accept':'application/json',
'accept-encoding': 'gzip, deflate, br',
'accept-language':'en-US,en;q=0.9',
'content-type': 'application/json',
'cookie': '_ga=GA1.2.1131534267.1668115448; _gid=GA1.2.1680305408.1668627447',
'origin': 'https://fakeyou.com',
'referer': 'https://fakeyou.com/',
'sec-ch-ua':'"Microsoft Edge";v="107", "Chromium";v="107", "Not=A?Brand";v="24"',
'sec-ch-ua-mobile': '?0',
'sec-ch-ua-platform': "Windows",
'sec-fetch-dest':'empty',
'sec-fetch-mode':'cors',
'sec-fetch-site': 'same-site',
'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.42'
}


data={
"inference_text": "how are you doing history go to fucking hell",
"tts_model_token": "TM:cc81anqyp6sd",
'uuid_idempotency_token': "ef8ac81c-8155-4fdc-970a-b16ebeecae34"
}

a=requests.post(url="https://api.fakeyou.com/tts/inference",json=data,headers=header,allow_redirects=True)
print(a.status_code)
print(a.text)
link="https://storage.googleapis.com/vocodes-public"
t=json.load(a.text)
if t['success'] :
  while True:
    b=requests.post(url="https://api.fakeyou.com/tts/job/"+t["inference_job_token"]).json()
    if b["state"]["status"] == "complete_success":
      link+=b["state"]['maybe_public_bucket_wav_audio_path']
      break
print(link)
import dall_e
dall_e.load_model()'''
import requests

'''header = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
session = requests.Session()
session.post('https://musicaldown.com/ar',headers=header)
cookies = requests.utils.cookiejar_from_dict(requests.utils.dict_from_cookiejar(session.cookies))
print(cookies)
session.post('https://musicaldown.com/ar',headers=header,cookies=cookies)
cookies = requests.utils.cookiejar_from_dict(requests.utils.dict_from_cookiejar(session.cookies))
tran=requests.post('https://musicaldown.com/ar/download',data=data,headers=header,cookies=cookies)
soup = Soup(tran.text,features="lxml")
div = soup.find_all("a", {"class": "download"})
print(div)
'''
from requests_html import HTMLSession,HTML
session = HTMLSession()
'''
r = session.post('https://musicaldown.com/ar')
secoundval,firstval=r.html.find('input')[1],r.html.find('input')[0]
name,value,url=secoundval.attrs["name"],secoundval.attrs['value'],firstval.attrs['name']
data={url:'https://www.tiktok.com/@zz5w/video/7053949687883205890',
name: value,
'verify': '1'
}
cookies=r.cookies
q=session.post('https://musicaldown.com/ar/download',data=data,cookies=cookies)
href=[i.attrs["href"]for i in q.html.find(".orange") if 'href' in i.attrs if 'tiktok' in i.attrs["href"]]
print(href)
a=session.get("https://sssinstagram.com")
cookies=a.cookies
data={
    "link":"https://www.instagram.com/p/BUNFyu5lhlU/",
    'token':''}
r=requests.post("https://sssinstagram.com/request",data=data,cookies=cookies)
print(r.text)'''
'''
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
image_data = requests.get(url, stream=True).raw
image = Image.open(image_data)
object_detector = pipeline('object-detection')
object_detector(image)'''
#print(findall(q.text))
#print(r.content)
'''msg=
قال محمد ابن عبدالسلام
                       يونس ماحد يقربه علي بالحرام 

يونس لاحضر للمجلس واحنا جلوس
                        من هيبته يفزوا له القوم قيام 

و لو حاولت تناخشه حتى بإبرة
                         كتمها بقلبه و جاك وقت المنام'
data={"msg":msg, 
'lang':'Zeina',
'source': 'ttsmp3'}
respone=requests.post("https://ttsmp3.com/makemp3_new.php",data=data)
respone=respone.json()
print(respone)'''''
'''
import base64
msg='هلا فيكم يا جماعة الخير معكم مرق من قناة مرق دوت كوم اليوم بنتكلم عن مواصفات المرق<'
data={
  "locale": "ar-AE",
"content": 
"""<voice name="ar-OM-AbdullahNeural">{}</voice>""".format(msg),
"ip": 
"212.36.220.84"
}
respone=requests.post("https://app.micmonster.com/restapi/create",data=data)
print(respone.text)
#a="{respone.text}"
file_content=base64.b64decode(respone.text)
with open("test.mp3","wb") as test:
 test.write(file_content)
 test.close()
from bs4 import BeautifulSoup as Soup
text="this is string"
textt=
print(bool(Soup(text, "html.parser").find()))
print(bool(Soup(textt, "html.parser").find()))

import re
t="Kim was born in 1890 into a rich family in the English city of Leeds."
print(re.search(t,"A. Kim was born in 1890 into a rich family in the English city of Leeds."))'''
gram='''Cybersecurity refers to the protection of computers, networks, and information systems from unauthorized access, theft, damage, or other cyber threats. With the increasing dependency on digital devices and the internet, cybersecurity has become a critical issue for individuals, businesses, and governments around the world. In this essay, we will discuss the importance of cybersecurity, its challenges, and some best practices to ensure effective cybersecurity.

Firstly, cybersecurity is essential because cyber threats can cause severe damage to individuals and organizations. Cyber attacks can result in the loss of sensitive data, financial loss, and reputational damage. For businesses, a cyber attack can lead to disruption of services, loss of customer trust, and legal liabilities. For governments, cyber attacks can impact national security and critical infrastructure such as power grids, transportation systems, and financial institutions. Therefore, effective cybersecurity measures are necessary to prevent cyber attacks and minimize their impact.

Secondly, the challenges in cybersecurity are numerous and constantly evolving. Cyber threats come in various forms, including phishing attacks, malware, ransomware, and distributed denial-of-service (DDoS) attacks. Cybercriminals are continually finding new ways to breach security systems, and therefore, cybersecurity measures must keep pace with these advancements. Additionally, cybersecurity requires cooperation and coordination between various stakeholders, including individuals, businesses, and governments. This can be challenging, especially given the varying interests and priorities of these stakeholders.

Despite the challenges, there are some best practices that individuals and organizations can follow to ensure effective cybersecurity. Firstly, it is essential to keep software and operating systems up to date with the latest security patches. These patches are released regularly to fix known vulnerabilities that could be exploited by cybercriminals. Secondly, individuals and organizations should use strong passwords and multi-factor authentication to protect their accounts. This will make it difficult for cybercriminals to gain unauthorized access to systems and networks. Thirdly, regular backups of data should be maintained to ensure that critical data can be recovered in case of a cyber attack. Lastly, individuals and organizations should stay vigilant and be aware of the latest cyber threats and their modus operandi. This will enable them to recognize and respond to potential cyber attacks promptly.

In conclusion, cybersecurity is a critical issue in the digital age. Cyber threats can cause significant damage to individuals, businesses, and governments. Effective cybersecurity measures are necessary to prevent cyber attacks and minimize their impact. Despite the challenges, individuals and organizations can follow best practices such as keeping software up to date, using strong passwords, maintaining regular backups, and staying vigilant to ensure effective cybersecurity.'''
'''data={
    "data":gram,
    "lang": "en",
     "mode": 1
}
get=session.get("https://www.paraphrase-online.com/",allow_redirects=True)
find=HTML(html=get.text)
find=find.find("meta")[3].attrs
found=find["content"]
d=get.cookies.get_dict()
d["x-csrf-token"]= found
header=session.headers
header["x-csrf-token"]= found
header['Content-type']='application/x-www-form-urlencoded; charset=UTF-8'
header['x-requested-with']= 'XMLHttpRequest'
phrasing=requests.post("https://www.paraphrase-online.com/phrasing",data=data,cookies=d,headers=header)
print(phrasing.text)
phrasing=phrasing.text.split(' ', 1)[1]
phrasing = phrasing.rsplit(" ", 1)[0]
''''''
from transformers import *

# models we gonna use for this tutorial
model_names = [
  "tuner007/pegasus_paraphrase",
  "Vamsi/T5_Paraphrase_Paws",
  "prithivida/parrot_paraphraser_on_T5", # Parrot
]

model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
  # tokenize the text to be form of a list of token IDs
  inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
  # generate the paraphrased sentences
  outputs = model.generate(
    **inputs,
    num_beams=num_beams,
    num_return_sequences=num_return_sequences,
  )
  # decode the generated sentences using the tokenizer to get them back to text
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)

sentence = "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences."

get_paraphrased_sentences(model, tokenizer, sentence, num_beams=10, num_return_sequences=10)

get_paraphrased_sentences(model, tokenizer, "To paraphrase a source, you have to rewrite a passage without changing the meaning of the original text.", num_beams=10, num_return_sequences=10)

tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

get_paraphrased_sentences(model, tokenizer, "paraphrase: " + "One of the best ways to learn is to teach what you've already learned")

# !pip install git+https://github.com/PrithivirajDamodaran/Parrot_Paraphraser.git'''

import openai

openai.api_key="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"
import sseclient
import urllib3

def open_stream(url, headers):
    """Get a streaming response for the given event feed using urllib3."""
    http = urllib3.PoolManager()
    return http.request('POST', url, preload_content=False, headers=headers,fields={"model": "text-davinci-003",
      "prompt": "What is Python?",
      "max_tokens": 100,
      "temperature": 0,
      "stream": True,})

if __name__ == '__main__':
    url = 'https://api.openai.com/v1/completions'
    headers = {'Accept': 'text/event-stream',"Authorization": "Bearer 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"}
    response = open_stream(url, headers)
    client = sseclient.SSEClient(response)
    stream = client.events()

    while True:
        event = next(stream)
        print(f"event: {event}")
#print(open['choices'][0]['message']['content'])
