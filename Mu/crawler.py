import requests
import re
import bs4
from bs4 import BeautifulSoup


# demo_url = "https://python123.io/ws/demo.html"


def print_zuihaodaxue():
    uinfo = []
    url = "http://www.zuihaodaxue.com/zuihaodaxuepaiming2018.html"
    html = get_html_text(url)
    fill_univ_list(uinfo, html)
    print_univ_list(uinfo, 20)


def get_soup(url, parser='html.parse'):
    return BeautifulSoup(get_html_text(url), parser)


def get_html_text(url):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except Exception:
        return ""


def fill_univ_list(ulist, html):
    soup = BeautifulSoup(html, 'html.parser')
    for tr in soup.find('tbody').children:
        if isinstance(tr, bs4.element.Tag):
            tds = tr('td')
            ulist.append([tds[0].string, tds[1].string, tds[2].string, tds[3].string])
    pass


def print_univ_list(ulist, num):
    tplp = "{0:^12}\t{1:{4}^12}\t{2:^12}\t{3:^12}"

    print(tplp.format('排名', '学校名称', '省市', '总分', chr(12288)))
    for i in range(num):
        u = ulist[i]
        print(tplp.format(u[0], u[1], u[2], u[3], chr(12288)))
    print("Suc" + str(num))


if __name__ == '__main__':
    print_zuihaodaxue()
