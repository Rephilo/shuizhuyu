import requests
import re
import bs4
import traceback
from bs4 import BeautifulSoup


# demo_url = "https://python123.io/ws/demo.html"
# url = "https://s.taobao.com/search?initiative_id=tbindexz_20170306&ie=utf8&spm=a21bo.2017.201856-taobao-item.2&sourceId=tb.index&search_type=item&ssid=s5-e&commend=all&imgfile=&q=switch&suggest=history_2&_input_charset=utf-8&wq=&suggest_query=&source=suggest"

def print_zuihaodaxue():
    uinfo = []
    url = "http://www.zuihaodaxue.com/zuihaodaxuepaiming2018.html"
    html = get_html_text(url)
    fill_univ_list(uinfo, html)
    print_univ_list(uinfo, 20)


def get_soup(url, parser='html.parse'):
    return BeautifulSoup(get_html_text(url), parser)


def get_html_text(url, code='utf-8'):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = code
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
    tplp = "{0:^12}\t{1:{4}^12}\t{2:^12}\t{3_tree:^12}"

    print(tplp.format('排名', '学校名称', '省市', '总分', chr(12288)))
    for i in range(num):
        u = ulist[i]
        print(tplp.format(u[0], u[1], u[2], u[3], chr(12288)))
    print("Suc" + str(num))


##############
def parse_page(ilt, html):
    try:
        plt = re.findall(r'\"view_price\":\"[\d.]*\"', html)
        tlt = re.findall(r'\"raw_title\":\".*?\"', html)

        for i in range(len(plt)):
            price = eval(plt[i].split(':')[i])
            title = eval(tlt[i].split(':')[i])
            ilt.append([price, title])
    except:
        print("")


def print_good_list(ilt):
    tplt = "{:4}\t{:8}\t{:6}"
    print(tplt.format("序号", "价格", "商品名称"))
    count = 0

    for g in ilt:
        count = count + 1
        print(tplt.format(count, g[0], g[1]))
    print("")


def get_taobao_good():
    goods = "switch"
    depth = 2
    start_url = "https://s.taobao.com/search?q=" + goods
    infoList = []
    for i in range(depth):
        try:
            url = start_url + '&s=' + str(44 * i)
            html = get_html_text(url)
            parse_page(infoList, html)
        except BaseException:
            continue
    print_good_list(infoList)


def get_stock_list(lst, stock_url):
    soup = BeautifulSoup(get_html_text(stock_url), 'html.parser')
    a = soup.find_all('a')
    for i in a:
        try:
            href = i.attrs['href']
            lst.append(re.findall(r'[s][hz]\d{6}', href)[0])
        except:
            continue
    pass


def get_stock_info(lst, stock_url, fpath):
    count = 0

    for stock in lst:
        url = stock_url + stock + ".html"
        html = get_html_text(url)
        try:
            if html == "":
                continue
            infoDict = {}
            soup = BeautifulSoup(html, 'html.parser')
            stockInfo = soup.find('div', attrs={'class': 'stock-bets'})

            name = stockInfo.find_all(attrs={'class': 'bets-name'})[0]
            infoDict.update({'股票名称': name.text.split()[0]})

            keyList = stockInfo.find_all('dt')
            valueList = stockInfo.find_all('dd')
            for i in range(len(keyList)):
                key = keyList[i].text
                val = valueList[i].text
                infoDict[key] = val

            count = count + 1
            print('\r当前进度：{:2f}%'.format(count * 100 / len(lst)))
        except:
            traceback.print_exc()
    pass


def get_stock_info():
    stock_list_url = 'https://quote.eastmoney.com/stocklist.html'
    stock_info_url = 'https://gupiao.baidu.com/stock/'
    output_file = '/'
    slist = []
    get_stock_list(slist, stock_list_url)
    get_stock_info(slist, stock_list_url, stock_info_url)


if __name__ == '__main__':
    get_taobao_good()
