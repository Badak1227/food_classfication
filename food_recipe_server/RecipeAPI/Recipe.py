import requests




# API 호출 함수
def fetch_recipes(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # 오류가 있을 경우 예외 발생
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"API 호출 중 오류 발생: {e}")
        return None

# 음식 이름과 재료로 필터링하는 함수
def filter_recipes(data, food_name, ingredients):
    recipes = data.get('COOKRCP01', {}).get('row', [])
    matching_recipes = []

    for recipe in recipes:
        if food_name not in recipe.get('RCP_NM', ''):
            continue

        for ingredient in ingredients:
            if ingredient not in recipe.get('RCP_PARTS_DTLS', ''):
                break
        else:
            matching_recipes.append(recipe)

    return matching_recipes