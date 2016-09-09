# encoding=utf-8
# web compare created for Xing at 2016/07/09
import webbrowser
from pandas import DataFrame
answer = []
with open('/Users/liuqianchao/Desktop/web/26_28超人.txt') as f:
    for item in f:
        if item.startswith('http'):
            url1 = item.split('	')[0]
            url2 = item.split('	')[1]
            url2 = url2.strip('\r\n')

            content = '''<frameset  rows=""><frameset cols="50%,50%"><frame src="''' +url1+'''"><frame src="'''+url2+'''"></frameset></frameset>'''
            with open('/Users/liuqianchao/Desktop/web/web.html','w') as wf:
                wf.write(content)
            webbrowser.open('file:///Users/liuqianchao/Desktop/web/web.html')
            answer.append(input('Please input this web\'s type:'))
reader = DataFrame(answer)
reader.columns = ['Type']
reader.to_csv('/Users/liuqianchao/Desktop/web/result.csv')