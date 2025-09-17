from ddgs import DDGS
import requests
import os

words = [
    "т банк логотип",
    "логотип т банк официальный",
    "логотип т банк скачать",
    "т банк новый логотип",
    "т банк фирменный стиль",
    "т банк айдентика",
    "т банк брендбук",
    "т банк на вывеске",
    "т банк отделение фото",
    "т банк банкомат логотип",
    "т банк на карте города",
    "т банк реклама",
    "т банк наружная реклама",
    "т банк уличная реклама",
    "т банк презентация",
    "т банк бизнес карточка",
    "т банк визитка",
    "т банк мобильное приложение",
    "т банк приложение логотип",
    "т банк продукты",
    "т банк дебетовая карта",
    "т банк кредитная карта",
    "т банк мобильный оператор",
    "т мобайл логотип",
    "т мобайл официальный логотип",
    "т мобайл фирменный стиль",
    "т мобайл айдентика",
    "т мобайл реклама",
    "т мобайл тарифы",
    "т мобайл приложение",
    "т банк инвестиции логотип",
    "т инвестиции логотип",
    "t bank logo",
    "t-bank branding",
    "t-mobile by tinkoff logo",
    "т банк желтый щит логотип",
    "т банк буква т логотип"
]


def download_duckduckgo_images(query, max_results=10, folder="images"):
    ddgs = DDGS()
    results = ddgs.images(keywords=query, max_results=max_results)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, image in enumerate(results):
        img_url = image.get("image")
        if img_url:
            try:
                response = requests.get(img_url, timeout=10)
                if response.status_code == 200:
                    with open(os.path.join(folder, f"{query}_{i+1}.jpg"), "wb") as f:
                        f.write(response.content)
                    print(f"Downloaded image {i+1}: {img_url}")
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")

# for i in range(len(words)):
download_duckduckgo_images('n', max_results=1)

