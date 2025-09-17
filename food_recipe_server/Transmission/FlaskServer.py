from flask import Flask, request, jsonify, render_template, url_for, session, redirect, send_file
from flask_cors import CORS
from Predict.PredictModel import *
from RecipeAPI.Recipe import filter_recipes
import os
import uuid

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Image as rImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.utils import ImageReader
import shutil
import requests
from PIL import Image
from io import BytesIO

def create_app(model, class_names, recipe_data):
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'template'))
    CORS(app, resources={r"/upload": {"origins": "*"}})

    def initialize_app():
        # 전역 변수 대신 Flask 컨텍스트에 저장
        app.config['MODEL'] = model
        app.config['CLASS_NAMES'] = class_names
        app.config['RECIPE_DATA'] = recipe_data
        app.secret_key = uuid.uuid4().hex
    initialize_app()

    @app.route('/')
    def home():
        return  render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload():
        image_data = request.files['image']
        ingredients_data = request.form.get('ingredients')
        ingredients = ingredients_data.replace(' ','').split(',')

        filename = f"{uuid.uuid4().hex}.jpg"

        image_data.save(f'./uploads/{filename}')
        image_url = url_for('uploaded_image', filename=filename)

        # 세션에 이미지 URL과 재료 데이터 저장
        session['image_url'] = image_url
        session['ingredients'] = ingredients

        # 리다이렉트
        return redirect(url_for('recipe'))

    @app.route('/uploads/<filename>', endpoint='uploaded_image')
    def uploaded_image(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/recipe')
    def recipe():

        ingredients = session.get('ingredients')
        image_url = session.get('image_url')
        image_path = f'./uploads/{image_url.split("/")[-1]}'

        food_name = predict_image(app.config['MODEL'], image_path, app.config['CLASS_NAMES'])
        session['food_name'] = food_name
        recipes = filter_recipes(app.config['RECIPE_DATA'], food_name, ingredients)

        result = []

        # 결과 출력
        if recipes:
            print(f"'{food_name}'와(과) {ingredients} 를 포함하는 레시피 목록")
            for idx, recipe in enumerate(recipes, start=1):
                # 만드는 과정 출력 (필드는 JSON 구조에 따라 다를 수 있음)

                steps = []
                for i in range(1, 21):  # 최대 20단계까지 확인
                    manual_key = f'MANUAL{i:02d}'
                    manual_img_key = f'MANUAL_IMG{i:02d}'

                    manual = recipe.get(manual_key, '').strip()
                    manual_img = recipe.get(manual_img_key, '').strip()

                    if manual:  # 조리 과정이 있으면 추가
                        steps.append({'description': manual, 'image': manual_img if manual_img else None})

                result.append({
                    'title': recipe['RCP_NM'],
                    'ingredients': recipe['RCP_PARTS_DTLS'],
                    'steps': steps
                })
        else:
            print(f"'{food_name}'와(과) 를 포함하는 레시피를 찾을 수 없습니다.")

        return render_template('recipes.html', recipes=result)



    @app.route('/generate_pdf', methods=['POST'])
    def generate_pdf():
        # 한글 폰트 등록
        pdfmetrics.registerFont(TTFont('NanumGothic', './template/3.NanumGothic.ttf'))

        to_pdf = request.get_json()

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        # 한글 폰트 적용된 스타일 생성
        styles.add(ParagraphStyle(name='KoreanTitle', fontName='NanumGothic', fontSize=18, spaceAfter=20))
        styles.add(ParagraphStyle(name='KoreanHeading2', fontName='NanumGothic', fontSize=14, spaceAfter=10))
        styles.add(ParagraphStyle(name='KoreanNormal', fontName='NanumGothic', fontSize=12, spaceAfter=10))

        story = []

        # 제목
        story.append(Paragraph(f"Recipe: {to_pdf['title']}", styles['KoreanTitle']))
        story.append(Spacer(1, 20))

        # 재료
        story.append(Paragraph("Ingredients:", styles['KoreanHeading2']))
        ingredients = to_pdf['ingredients'].split(", ")
        for ingredient in ingredients:
            ingredient = ingredient.replace('<br>', '')
            story.append(Paragraph(f"- {ingredient}", styles['KoreanNormal']))
        story.append(Spacer(1, 20))

        # 단계
        story.append(Paragraph("Steps:", styles['KoreanHeading2']))
        story.append(Spacer(1, 10))

        for idx, step in enumerate(to_pdf['steps'], start=1):
            # 단계 설명
            story.append(Paragraph(f"Step {step['description']}", styles['KoreanNormal']))
            story.append(Spacer(1, 10))

            # 이미지 삽입
            if step['image']:
                try:
                    response = requests.get(step['image'], timeout=5)
                    response.raise_for_status()

                    image = Image.open(BytesIO(response.content))
                    image.thumbnail((300, 300))  # 이미지 크기 조정

                    # PIL 이미지를 BytesIO로 변환
                    img_buffer = BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)

                    img = rImage(img_buffer, width=150, height=150)
                    story.append(img)
                    story.append(Spacer(1, 20))
                except Exception as e:
                    story.append(Paragraph(f"Failed to load image: {e}", styles['KoreanNormal']))
                    story.append(Spacer(1, 10))

        story.append(Spacer(1, 10))

        # PDF 생성
        doc.build(story)
        buffer.seek(0)
        return send_file(buffer, mimetype='application/pdf', as_attachment=True, download_name=f"{to_pdf['title']}.pdf")


    @app.route('/eval', methods=['POST'])
    def eval():
        data = request.get_json()
        response = data.get('response')

        if response == 'yes':
            food_name = session.get('food_name')
            target_folder = os.path.join(f'./dataset', food_name)

            os.makedirs(target_folder, exist_ok=True)

            image_url = session.get('image_url')
            image_path = f'./uploads/{image_url.split("/")[-1]}'

            destination_path = os.path.join(target_folder, os.path.basename(image_path))
            shutil.move(image_path, destination_path)

            return jsonify({"message": f"Thank you for confirming!"}), 200
        elif response == 'no':
            image_url = session.get('image_url')
            image_path = f'./uploads/{image_url.split("/")[-1]}'

            os.remove(image_path)
            return jsonify({"message": f"Thank you for confirming!"}), 200
        else:
            # 잘못된 입력에 대한 처리
            return jsonify({"error": "Invalid response. Please send 'yes' or 'no'."}), 400

    return app