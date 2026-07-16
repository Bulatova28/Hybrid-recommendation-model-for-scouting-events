import pandas as pd
import numpy as np
import gradio as gr
import joblib
import pickle
import math
import hashlib
import scipy.stats as st
import torch

from torch.utils.data import Dataset, DataLoader

from dlrm_model import ScoutDLRM
from preprocessing_dlrm import SentenceTransformerEmbedder, CyclicEncoder, convert_to_int
from datetime import datetime, date

from scout_dataset import ScoutDataset, scout_collate_fn


print("--- [0] Початок завантаження системи... ---")

event_df = pd.read_pickle('event_df.pkl')
transformer = joblib.load('feature_transformer_gemma.pkl')

warmup_df = event_df.head(1).copy()
warmup_df['age'] = 18
warmup_df['category'] = 'скаут'
warmup_df['interests'] = 'мандрівництво'
warmup_df['start date'] = pd.to_datetime(warmup_df['start date'])
warmup_df['end date'] = pd.to_datetime(warmup_df['end date'])
warmup_df['duration'] = 1
warmup_df['season'] = 1

print("--- [0] Прогріваємо мовну модель (це може зайняти до 1 хв)... ---")
try:
    _ = transformer.transform(warmup_df)
    print("--- [0] Успіх! Модель готова. ---")
except Exception as e:
    print(f"--- [!] Попередження при прогріві: {e} ---")

dlrm_scout_model = ScoutDLRM(
    embed_dim=32, 
    embed_table_size=[6, 8, 4], 
    bot_mlp_size=[8, 128, 64, 32],
    text_mlp_size=[256, 64, 32],
    top_mlp_size=[42, 16, 5]
)
dlrm_scout_model.load_state_dict(torch.load('scout_dlrm_dyploma_gemma.pth', map_location='cpu'))
dlrm_scout_model.eval()


def get_deterministic_rating(name):
    hash_val = int(hashlib.md5(str(name).encode('utf-8')).hexdigest(), 16)
    return round(4.5 + (hash_val % 6) / 10.0, 1)
    

def get_scout_category(age: int):
    if 6 <= age <= 10:
        return 'кабскаут'
    elif 11 <= age <= 15:
        return 'скаут'
    elif 16 <= age <= 22:
        return 'ровер'
    else:
        return 'лідер'


def calculate_age_and_cat(dob_str):
    try:
        dob = datetime.strptime(dob_str, "%d.%m.%Y").date()
        today = date.today()
        
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        
        if age < 0:
            return gr.update(), gr.update()
            
        category = get_scout_category(age)
        return age, category
    except (ValueError, TypeError):
        return gr.update(), gr.update()


def update_cat_by_age(age):
    return get_scout_category(age)


def get_scout_recommendation(age: int, category: str, interests: list[str]):
    print(f"\n--- [1] Старт! Вік: {age}, Категорія: {category} ---")
    if not category:
        category = get_scout_category(age)

    current_date = datetime.now()

    filtered_event_df = event_df.loc[(event_df['start date']>= current_date) & ((age >= event_df['min age']) & (age <= event_df['max age']))].copy()

    print(f"--- [2] Фільтрація завершена. Знайдено заходів: {len(filtered_event_df)} ---")

    if filtered_event_df.empty:
        print("--- [!] Заходів не знайдено, вихід ---")
        return pd.DataFrame({"Повідомлення": ["На жаль, актуальних заходів не знайдено"]})

    filtered_event_df['age'] = age
    filtered_event_df['category'] = category
    filtered_event_df['interests'] = ', '.join(interests)

    print("--- [3] Починаю трансформацію (SentenceTransformer)... ---")

    x_prep = transformer.transform(filtered_event_df)
    print("--- [4] Трансформація завершена! ---")
    dummy_y = pd.Series([0] * len(filtered_event_df))

    inference_ds = ScoutDataset(x_prep, dummy_y)
    inference_loader = DataLoader(dataset = inference_ds, 
                                  batch_size = len(filtered_event_df),
                                  shuffle = False,
                                  collate_fn = scout_collate_fn)
    
    x_dense, x_text, offsets, indices, _ = next(iter(inference_loader))

    print("--- [5] Запуск DLRM моделі... ---")

    with torch.no_grad():
        weighted_avg, _ = dlrm_scout_model(x_dense, x_text, indices, offsets)

    print("--- [6] Прогноз отримано! ---")

    filtered_event_df['predicted_rating'] = weighted_avg.numpy()
    filtered_event_df['relevance'] = (filtered_event_df['predicted_rating'] / 5.0) * 100
    filtered_event_df['relevance'] = filtered_event_df['relevance'].clip(upper=100, lower=0).astype(int)

    top = filtered_event_df.sort_values(by='predicted_rating', ascending=False).head(6).copy()
    top = top.reset_index(drop=True)

    top['start date'] = pd.to_datetime(top['start date']).dt.strftime('%d.%m.%Y')
    top['end date'] = pd.to_datetime(top['end date']).dt.strftime('%d.%m.%Y')

    if 'avg_rating' not in top.columns:
        top['avg_rating'] = top['name'].apply(get_deterministic_rating)

    final_res = top[['name', 'description', 'type', 'start date', 'end date', 'price', 'currency', 'image_url', 'relevance', 'avg_rating']]
    
    print("--- [7] Результат сформовано! Ось перші рядки: ---")
    print(final_res.head(2))

    return final_res


css = """
gradio-app, .gradio-container, .main, .panel, .wrap, .prose {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.prose a, .prose a svg, h1 a, h1 svg, h2 a, h2 svg, .icon-button { 
    display: none !important; 
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
}

* { animation: none !important; }
.pending, .generating, .translucent, .opacity-50 { opacity: 1 !important; }

body, .gradio-container, span, p, label { font-size: 22px !important; }
h1 { font-size: 30px !important; }
h2 { font-size: 26px !important; }
h3 { font-size: 22px !important; }

#login-container { 
    max-width: 500px !important; 
    margin: 80px auto !important; 
    padding: 40px !important; 
    background: white !important; 
    border-radius: 20px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1) !important; 
    border: 1px solid #f0f0f0 !important;
}

.block .wrap, 
input[type="number"], 
input[type="text"], 
input[type="email"], 
input[type="password"] {
    border: 2px solid #e0e0e0 !important;
    border-radius: 10px !important;
    background-color: white !important;
    min-height: 55px !important;
    height: 65px !important; 
    box-sizing: border-box !important;
    margin: 0 !important; 
}

input[type="number"], 
input[type="text"], 
input[type="email"], 
input[type="password"] {
    padding-left: 15px !important;
}

.custom-purple-label span[data-testid="block-info"],
.custom-purple-label span[data-testid="block-info"] * {
    color: #4A148C !important;
    font-size: 22px !important;
    font-weight: 600 !important;
}

.custom-purple-label input,
.custom-purple-label .wrap-inner,
.custom-purple-label .wrap-inner *,
.custom-purple-label .single-select, 
.custom-purple-label .single-select *,
.custom-purple-label .token,
.custom-purple-label .token *,
.custom-purple-label .options,
.custom-purple-label .options * {
    font-size: 22px !important; 
    color: black !important;
}

.wrap-inner, .secondary-wrap { 
    border: none !important; 
    box-shadow: none !important; 
    background: transparent !important;
    min-height: 100% !important; 
}

.tabs { border: none !important; background: transparent !important; }
.tab-nav { border-bottom: 2px solid #e0e0e0 !important; margin-bottom: 30px !important; padding-bottom: 10px !important; background: transparent !important; }

button[role="tab"], 
button[role="tab"] span,
button[role="tab"] * {
    font-size: 22px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
}

button[role="tab"][aria-selected="true"],
button[role="tab"][aria-selected="true"] span,
button[role="tab"][aria-selected="true"] * {
    color: #4A148C !important;
    border-bottom: 4px solid #4A148C !important;
}

#sort-block span[data-testid="block-info"] {
    color: #4A148C !important;
    font-size: 22px !important;
    font-weight: 600 !important;
    padding-bottom: 10px
}

#sort-block {
    --block-border-width: 0px !important;
    --block-background-fill: white !important;
    border: none !important;
    box-shadow: none !important;
    background: white !important;
    background-color: white !important;
}

#sort-block .wrap, 
#sort-block fieldset, 
#sort-block .form {
    border: none !important;
    background: white !important;
    background-color: white !important;
    box-shadow: none !important;
}

#sort-block input[type="radio"] { appearance: radio !important; width: 24px !important; height: 24px !important; accent-color: #4A148C !important; cursor: pointer; }
#sort-block label.radio .radio-icon, #sort-block label.radio .radio-bg { display: none !important; }

.submit-btn-row { display: flex !important; justify-content: center !important; margin-top: 40px !important; margin-bottom: 20px !important; }
.primary-btn { background: #4A148C !important; color: white !important; font-weight: bold !important; font-size: 22px !important; max-width: 300px !important; width: 100% !important; height: 60px !important; margin: 0 auto !important; border: none !important; }

.cards-grid { 
    display: grid; 
    grid-template-columns: repeat(3, 460px) !important; 
    justify-content: center !important; 
    gap: 30px !important; 
    margin-top: 10px !important; 
    margin-bottom: 60px !important; 
}

.scout-card { 
    border: 1px solid #e0e0e0; 
    border-radius: 15px; 
    overflow: hidden; 
    background: white; 
    box-shadow: 0 6px 12px rgba(0,0,0,0.08); 
    display: flex; 
    flex-direction: column; 
    min-height: 650px; 
    transition: transform 0.3s ease, box-shadow 0.3s ease !important;
}

.scout-card:hover {
    transform: translateY(-10px) scale(1.01) !important; 
    box-shadow: 0 15px 35px rgba(74, 20, 140, 0.2) !important; 
    cursor: pointer !important; 
    z-index: 10 !important; 
}

.scout-card-img-real {
    width: 100% !important;
    height: 320px !important;
    object-fit: cover !important; 
    display: block !important;
    border-bottom: 1px solid #e0e0e0 !important; 
}

.scout-card-img-placeholder { 
    height: 320px; 
    background-color: #4A148C; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    color: white; 
    font-weight: bold; 
}

.scout-card-body { padding: 20px; display: flex; flex-direction: column; flex-grow: 1; }
.scout-card-title { font-size: 30px !important; font-weight: bold; color: #4A148C; text-align: center; margin-bottom: 15px; }
.scout-card-info { font-size: 22px !important; color: #444; margin-bottom: 8px; text-align: left; }
.scout-card-description { font-size: 22px !important; color: #666; margin-top: 15px; text-align: justify; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; }

.clean-html, 
.clean-html.block,
.clean-html .wrap {
    border: none !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

#results-layout {
    border: none !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
}

#sort-sidebar {
    border: none !important;
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    width: 300px !important;
    min-width: 300px !important;
    max-width: 300px !important;
    flex: 0 0 300px !important; 
    margin-top: 20px !important; 
}

#sort-block label.radio,
#sort-block label {
    display: flex !important;
    width: 100% !important;          
    min-width: 100% !important;      
    margin-bottom: 12px !important;  
    margin-right: 0 !important;      
    box-sizing: border-box !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

#sort-block,
#sort-block .wrap,
#sort-block .wrap-inner,
#sort-block div[data-testid="checkbox-group"],
#sort-block div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    height: auto !important;      
    max-height: none !important;   
    overflow: visible !important;  
}

.nuke-loaders .generating,
.nuke-loaders .progress,
.nuke-loaders .progress-text,
.nuke-loaders [data-testid="progress-text"],
.nuke-loaders .eta-bar,
.nuke-loaders .loader,
.nuke-loaders .wrap.pending::before,
.nuke-loaders .wrap.pending::after {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    position: absolute !important;
    z-index: -100 !important;
    pointer-events: none !important;
}

.nuke-loaders .translucent,
.nuke-loaders .pending {
    opacity: 1 !important;
}

.no-scroll-header, 
.no-scroll-header .wrap,
.no-scroll-header .prose {
    height: auto !important; 
    min-height: 0 !important;
    max-height: none !important; 
    overflow: visible !important; 
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.login-btn-row {
    display: flex !important;
    gap: 15px !important;
    margin-top: 25px !important;
    width: 100% !important;
}

.login-btn-row > * {
    flex: 1 !important; 
    min-width: 0 !important;
}

.secondary-btn {
    background: #4A148C !important; 
    color: white !important;        
    border: 2px solid #4A148C !important;
    font-weight: bold !important;
    font-size: 22px !important;
    height: 60px !important;
    margin: 0 !important;
    transition: all 0.3s ease !important;
}

.secondary-btn:hover,
.secondary-btn:active {
    background: white !important;
    background-color: white !important;
    color: #4A148C !important;
}

#login-container .block,
#login-container .form,
#login-container fieldset,
.custom-purple-label.block,
.custom-purple-label .form,
.custom-purple-label fieldset {
    border: none !important;
    background: white !important;
    background-color: white !important;
    box-shadow: none !important;
    --block-border-width: 0px !important;
    --form-border-width: 0px !important;
    --block-background-fill: white !important;
    --background-fill-primary: white !important;
    --background-fill-secondary: white !important;
}

.custom-purple-label .wrap {
    background: white !important;
    background-color: white !important;
}

#login-container .custom-purple-label {
    padding-left: 0 !important;
    padding-right: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
}

#login-container .custom-purple-label .wrap,
#login-container .custom-purple-label .form {
    width: 100% !important;
    max-width: 100% !important;
}

#login-container input:-webkit-autofill,
#login-container input:-webkit-autofill:hover, 
#login-container input:-webkit-autofill:focus, 
#login-container input:-webkit-autofill:active {
    -webkit-box-shadow: 0 0 0 1000px #fcf4ff inset !important; 
    -webkit-text-fill-color: black !important; 
    transition: background-color 5000s ease-in-out 0s;
}

#login-container .custom-purple-label input:focus,
#login-container .custom-purple-label input:active {
    background-color: #fcf4ff !important;
}

#tab_profile .custom-purple-label input:focus,
#tab_profile .custom-purple-label input:active,
#tab_profile .custom-purple-label .wrap:focus-within {
    background-color: white !important;
}

.primary-btn {
    border: 2px solid #4A148C !important; 
    transition: all 0.3s ease !important; 
}

.primary-btn:hover,
.primary-btn:active {
    background: white !important;
    background-color: white !important;
    color: #4A148C !important; 
}

.no-loader .generating,
.no-loader .progress,
.no-loader .progress-text,
.no-loader .eta-bar,
.no-loader .progress-level,
.no-loader svg,
.no-loader .wrap.pending::before,
.no-loader .wrap.pending::after,
.no-loader .loader {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    width: 0 !important;
    position: absolute !important;
    pointer-events: none !important;
}

.no-loader.translucent, 
.no-loader .translucent, 
.no-loader .opacity-50 { 
    opacity: 1 !important; 
}

"""


def update_cat_by_age(age):
    if 6 <= age <= 10: return 'кабскаут'
    elif 11 <= age <= 15: return 'скаут'
    elif 16 <= age <= 22: return 'ровер'
    return 'лідер'


def generate_cards_html(df):
    if df.empty or "Повідомлення" in df.columns:
        return "<h3 style='text-align: center; margin-top: 20px;'>На жаль, за вашими параметрами нічого не знайдено</h3>"
    
    html = "<div class='cards-grid'>"
    has_relevance = 'relevance' in df.columns
    for _, row in df.iterrows():
        name = row.get('name', 'Без назви')
        event_type = row.get('type', 'Загальний захід')
        dates = f"{row.get('start date', '')} — {row.get('end date', '')}"
        price_val = row.get('price')
        currency = row.get('currency', '')
        description = row.get('description', 'Опис відсутній')

        relevance = row.get('relevance', 0)
        avg_rating = row.get('avg_rating', 4.8)
        
        price_str = f"{price_val} {currency}" if pd.notna(price_val) and str(price_val).strip() else "Безкоштовно"
        
        img_url = row.get('image_url', '') 
        
        if pd.isna(img_url) or not str(img_url).strip() or str(img_url) == 'nan':
            img_html = "<div class='scout-card-img-placeholder'>⚜️ Scoutify Event</div>"
        else:
            img_html = f"<img src='{img_url}' class='scout-card-img-real' alt='{name}'>"
        
        full_stars = int(avg_rating)
        empty_stars = 5 - full_stars
        
        star_gold = "<span style='color: #FFC107; font-size: 26px; line-height: 1;'>★</span>"
        star_grey = "<span style='color: #E0E0E0; font-size: 26px; line-height: 1;'>★</span>"
        
        stars_html = (star_gold * full_stars) + (star_grey * empty_stars)

        relevance_html = ""
        if has_relevance:
            relevance = int(row['relevance'])
            relevance_html = f"<span style='font-size: 16px; vertical-align: middle; background: #e8eaf6; color: #4A148C; padding: 4px 10px; border-radius: 12px; font-weight: bold; margin-left: 5px;'>{relevance}%</span>"

        html += f"""
        <div class='scout-card'>
            {img_html}
            <div class='scout-card-body'>
                <div class='scout-card-title'>
                    {name} {relevance_html}
                </div>
                
                <div class='scout-card-info'>
                    🏷️ <b>Тип:</b> {event_type}
                </div>
                
                <div class='scout-card-info'>
                    📅 <b>Дати:</b> {dates}
                </div>
                
                <div class='scout-card-info'>
                    💰 <b>Вартість:</b> {price_str}
                </div>

                <div class='scout-card-description'>
                    {description}
                </div>

                <div style='display: flex; justify-content: flex-end; align-items: center; margin-top: auto; padding-top: 15px;'>
                    <span style='font-size: 20px; font-weight: bold; color: #4A148C; margin-right: 8px;'>{avg_rating:.1f}</span>
                    <span style='font-size: 18px; letter-spacing: 2px;'>{stars_html}</span>
                </div>
                
            </div>
        </div>
        """
    html += "</div>"
    return html


with gr.Blocks(theme=gr.themes.Default(primary_hue="purple", secondary_hue="indigo"), css=css) as demo:
    
    with gr.Column(elem_id="login-container") as login_box:
        gr.HTML("<h1 style='text-align: center; color: #4A148C; font-weight: bold; margin-bottom: 20px;'>⚜️ Scoutify ⚜️</h1>", elem_classes="no-scroll-header")
        
        email = gr.Textbox(label='Електронна пошта', placeholder='scout@example.com', max_lines=1, elem_classes="custom-purple-label")
        password = gr.Textbox(label='Пароль', type='password', elem_classes="custom-purple-label")
        
        with gr.Row(elem_classes="login-btn-row"):
            btn_login = gr.Button('Увійти', elem_classes="secondary-btn")
            btn_register = gr.Button('Зареєструватися', elem_classes="secondary-btn")

    with gr.Column(visible=False) as main_app:
        with gr.Tabs(elem_id="main-tabs") as tabs_obj:
            with gr.Tab('Головна'):
                with gr.Column(elem_classes="nuke-loaders"):
                    welcome_msg = gr.HTML(
                        "<div style='border: 2px solid #e0e0e0; border-radius: 12px; padding: 12px 25px; background-color: white; width: fit-content; margin: 0 auto;'>"
                        "<h3 style='text-align: center; margin: 0px;'>Вітаємо! Будь ласка, заповніть дані у Профілі для отримання рекомендацій</h3>"
                        "</div>", 
                        elem_classes="no-scroll-header no-loader" 
                    )
                    
                    recom_header = gr.HTML(value="", visible=False, elem_classes="no-scroll-header no-loader") 
                    main_results = gr.HTML(value="", visible=False, elem_classes="clean-html no-loader")       
                    popular_header = gr.HTML(value="", elem_classes="no-scroll-header no-loader") 
                    popular_results = gr.HTML(value="", elem_classes="clean-html no-loader")       

            with gr.Tab("Профіль", id="tab_profile"):
                gr.HTML("<div style='padding-top: 10px;'><h3 style='text-align: left; color: #333; margin-bottom: 5px;'>Налаштування профілю користувача</h3></div>")
                user_email_display = gr.HTML(value="", elem_classes="no-scroll-header")
                results_state = gr.State()

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            dob_input = gr.Textbox(label="Дата народження", placeholder="ДД.ММ.РРРР", max_lines=1, elem_classes="custom-purple-label")
                            age_input = gr.Number(label="Ваш вік", value=14, elem_classes="custom-purple-label")
                            cat_input = gr.Dropdown(
                                choices=['кабскаут', 'скаут', 'ровер', 'лідер', 'волонтер'], 
                                label="Категорія",
                                value='ровер',
                                elem_classes="custom-purple-label"
                            )

                    with gr.Column(scale=1):
                        interests_input = gr.Dropdown(
                            choices=['екологія', 'природа', 'дослідження', 'виживання', 'управління', 'вузли', 'медицина', 'картографія', 'кемпінг', 
                                     'скелелазіння', 'кулінарія', 'плавання', 'стрільба з лука', 'спортивне орієнтування', 'веломандрівки', 'полювання'],
                            multiselect=True, max_choices=3, label="Ваші інтереси (оберіть до 3-х)", elem_classes="custom-purple-label"
                        )
                
                dob_input.change(fn=calculate_age_and_cat, inputs=dob_input, outputs=[age_input, cat_input], show_progress="hidden")
                age_input.change(fn=update_cat_by_age, inputs=age_input, outputs=cat_input, show_progress="hidden")

                with gr.Row(elem_classes="submit-btn-row"):
                    btn_confirm = gr.Button("Підтвердити", variant="primary", elem_classes="primary-btn")

                prof_recom_header = gr.HTML(value="", elem_classes="no-scroll-header")

                with gr.Row(visible=False, elem_id="results-layout", elem_classes="no-loader") as results_layout_row:
                    with gr.Column(scale=1, min_width=300, elem_id="sort-sidebar"):
                        sort_input = gr.Radio(
                            choices=['За релевантністю', 'За датою'],
                            value='За релевантністю',
                            label="Сортування результатів",
                            elem_id="sort-block"
                        )
                    
                    with gr.Column(scale=3):
                        prof_results = gr.HTML(value="", elem_classes="no-loader clean-html")

    def login_action(user_email):
        email_html = f"<div style='text-align: left; margin-bottom: 80px;'><span style='color: #4A148C; font-size: 22px; text-decoration: underline; font-weight: 500;'>{user_email}</span></div>"
        pop_header = "<h1 style='text-align: center; color: #4A148C; margin-top: 5px; margin-bottom: 22px;'>🔥 Найпопулярніші заходи 🔥</h1>"
        pop_df = event_df.head(6).copy()

        if 'avg_rating' not in pop_df.columns:
            pop_df['avg_rating'] = pop_df['name'].apply(get_deterministic_rating)
        
        pop_df['start date'] = pd.to_datetime(pop_df['start date']).dt.strftime('%d.%m.%Y')
        pop_df['end date'] = pd.to_datetime(pop_df['end date']).dt.strftime('%d.%m.%Y')
        pop_html = generate_cards_html(pop_df)
        
        return gr.update(visible=False), gr.update(visible=True), email_html, pop_header, pop_html

    btn_login.click(
        fn=login_action, 
        inputs=[email], 
        outputs=[login_box, main_app, user_email_display, popular_header, popular_results],
        show_progress="hidden" 
    )

    btn_register.click(
        fn=login_action, 
        inputs=[email], 
        outputs=[login_box, main_app, user_email_display, popular_header, popular_results],
        show_progress="hidden" 
    )
    

    def handle_sort(df_records, sort_type):
        if not df_records: 
            return ""
        
        temp_df = pd.DataFrame(df_records)
        
        if sort_type == 'За датою':
            temp_df['_temp_date'] = pd.to_datetime(temp_df['start date'], format='%d.%m.%Y')
            temp_df = temp_df.sort_values(by='_temp_date', ascending=True).drop(columns=['_temp_date'])
        
        return generate_cards_html(temp_df)

    def handle_confirm(age, category, interests, sort_type):
        if not interests:
            return [
                gr.update(), gr.update(visible=False), gr.update(visible=False), "", 
                gr.update(visible=False), "<h3 style='text-align: center; color: red;'>⚠️ Оберіть інтерес!</h3>", []
            ]
        
        try:
            res_df = get_scout_recommendation(age, category, interests)
            df_records = res_df.to_dict('records')
            initial_html = handle_sort(df_records, sort_type)
            header_html = "<h1 style='text-align: center; color: #4A148C; margin-top: 5px; margin-bottom: 22px;'>✨ Рекомендовано ✨</h1>"
            welcome_update = f"""
            <div style='border: 2px solid #e0e0e0; border-radius: 12px; padding: 12px 25px; background-color: white; width: fit-content; margin: 0 auto;'>
                <h3 style='text-align: center; margin: 0px;'>Профіль успішно оновлено!</h3>
            </div>
            """
            
            return [
                welcome_update,                               
                gr.update(value=header_html, visible=True),   
                gr.update(value=initial_html, visible=True),  
                header_html,                                  
                gr.update(visible=True),                      
                initial_html,                                 
                df_records                                    
            ]
        
        except Exception as e:
            error_msg = f"<h3 style='text-align: center; color: red;'>❌ Помилка: {e}</h3>"
            return [gr.update(), gr.update(visible=False), gr.update(visible=False), "", gr.update(visible=False), error_msg, []]

    btn_confirm.click(
        fn=handle_confirm,
        inputs=[age_input, cat_input, interests_input, sort_input],
        outputs=[welcome_msg, recom_header, main_results, prof_recom_header, results_layout_row, prof_results, results_state],
        show_progress=True 
    )

    sort_input.change(
        fn=lambda df, st: (handle_sort(df, st), handle_sort(df, st)),
        inputs=[results_state, sort_input],
        outputs=[main_results, prof_results]
    )

demo.queue().launch()