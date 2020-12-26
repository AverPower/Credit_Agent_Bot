import pandas as pd
import numpy as np
import os
import pickle
import datetime
import time
from telegram import Bot
from telegram import Update
from telegram import ParseMode
from telegram import InlineKeyboardButton
from telegram import InlineKeyboardMarkup
from telegram.ext import CallbackContext
from telegram.ext import Filters
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler
from telegram.ext import CallbackQueryHandler
from telegram import InlineQueryResultArticle, ParseMode, InputTextMessageContent
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

TOKEN =''
      
user_info = dict()
last_action = ""

def send_welcome(update, context):
    user = update.message.from_user
    chat_id = update.effective_message.chat_id
    context.bot.send_message(
            chat_id=chat_id,
            parse_mode = 'MarkDown',
            text=f"""Добрый день, *{user.first_name} {user.last_name}*!
Вас приветствует **Credit Agent Bot** - Ваш помощник для получения лучшего предложения по потребительскому кредиту!"""
    )
    time.sleep(3)
    context.bot.send_message(
            chat_id=chat_id,
            parse_mode = 'MarkDown',
            text=f"""Суть данного проекта заключается в следующем:
Пользователь (то есть Вы) желаете получить потребительский кредит. 
В данном чате вы отвечаете на простые анкетные вопросы."""
    )
    time.sleep(3)
    context.bot.send_message(
            chat_id=chat_id,
            parse_mode = 'MarkDown',
            text=f"""На основе предоставленной Вами информации формируются анкеты в ряд банков *(в данном демо их два)*. 
На основе обученных моделей кредитного скоринга каждый банк принимает по Вам решение - стоит ли выдавать Вам кредит - и формирует базовое предложение.
""" )
    time.sleep(3)
    context.bot.send_message(
            chat_id=chat_id,
            parse_mode = 'MarkDown',
            text=f"""Наш бот собирает заявки от банков, готовых выдать кредит, и проводит голландский аукцион, где последовательно участники делают предложения с меньшей общей стоимостью кредита, варьируя процентную ставку, первоначальный взнос и срок *(в данном демо только процентную ставку)*. 
Победитель выдает кредит на условиях, сформированных в процессе проведения аукциона."""
    )
    time.sleep(3)
    keyboard = []
    keyboard.append([InlineKeyboardButton('Да', callback_data = 'Start_yes')])
    keyboard.append([InlineKeyboardButton('Нет', callback_data = 'Start_no')])

    context.bot.send_message(chat_id=chat_id, text='Ну что, Вы готовы начать?', reply_markup=InlineKeyboardMarkup(keyboard))
    

def keyboard_callback_handler(update, context):
        
    query = update.callback_query
    chat_id = update.effective_message.chat_id
    data = query.data
    
    if data.startswith('Start_'):
        if data == "Start_yes":
            keyboard = []
            keyboard.append([InlineKeyboardButton('Мужчина', callback_data = 'gender_male')])
            keyboard.append([InlineKeyboardButton('Женщина', callback_data = 'gender_female')])
            context.bot.send_message(chat_id=chat_id, text='Укажите ваш пол:', reply_markup=InlineKeyboardMarkup(keyboard))
        else:
            context.bot.send_message(chat_id=chat_id, text=f'Жаль, приходите еще...')
    
    elif data.startswith('gender_'):
        user_info['CODE_GENDER'] = 1 if data == 'gender_male' else 0
        keyboard = []
        keyboard.append([InlineKeyboardButton('Да', callback_data = 'car_yes')])
        keyboard.append([InlineKeyboardButton('Нет', callback_data = 'car_no')])
        context.bot.send_message(chat_id=chat_id, text='Имеется ли у Вас автомобиль?', reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif data.startswith('car_'):
        user_info['FLAG_OWN_CAR'] = 1 if data == 'car_yes' else 0
        keyboard = []
        keyboard.append([InlineKeyboardButton('Да', callback_data = 'realty_yes')])
        keyboard.append([InlineKeyboardButton('Нет', callback_data = 'realty_no')])
        context.bot.send_message(chat_id=chat_id, text='Имеется ли у Вас недвижимость?', reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif data.startswith('realty_'):
        user_info['FLAG_OWN_REALTY'] = 1 if data == 'realty_yes' else 0
        global last_action
        last_action = 'realty'
        context.bot.send_message(chat_id=chat_id, text='Укажите Ваш среднемесячный доход:')
        
        
    elif data.startswith('status_'):
        stat = data.split('_')[1]
        cats = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
        cats_call = ['married', 'single', 'civil', 'separated', 'widow']
        for cat, cc in zip(cats_call, cats):
            user_info[cc] = 1 if stat == cat else 0
        context.bot.send_message(chat_id=chat_id, text=f"""Поздравляем, вопросы закончились! 
На основе Ваших данных сформированы анкеты и отправлены в банки.
На данный момент Ваши данные проверяются""")
        time.sleep(2)
        banks_names = ['BangBank', 'Smirnoff Bank'] 
        banks_base_percentage = [0.2, 0.15]
        decisions = get_banks_decisions(user_info)
        if np.sum(decisions) == 2:
            context.bot.send_message(chat_id=chat_id, text=f"""Хорошие новости! Несколько банков готовы сделать Вам предложение 
В данный момент проходит аукцион...""")
            winner, percent = auction_starter(user_info, banks_base_percentage)
        else:
            context.bot.send_message(chat_id=chat_id, text=f"""Хорошие новости! Мы нашли банк, который сформировал для Вас предложение.""")
            for i in range(len(decisions)):
                if decisions[i] == 1:
                    winner = i
            percent = banks_base_percentage[winner]
        time.sleep(2)    
        month_pay = calculate_credit_value(np.exp(user_info['AMT_CREDIT']) / 8, percent)
        context.bot.send_message(chat_id=chat_id, text=f"""Итак, банк {banks_names[winner]} предлагает Вам кредит под {percent * 100 :.2f} процентов с суммой ежемесячного платежа {int(month_pay)} на 36 месяцев""")
        
def feedback(update, context):
    chat_id = update.effective_message.chat_id
    global last_action
    text = update['message']['text']
    if last_action == 'realty':
        user_info['AMT_INCOME_TOTAL'] = np.log(int(text) * 8 / 5)
        last_action = 'income'
        context.bot.send_message(chat_id=chat_id, text='Какую сумму Вы бы хотели получить?')
    elif last_action == 'income':
        user_info['AMT_CREDIT'] = np.log(int(text) * 8)
        last_action = 'kids'
        context.bot.send_message(chat_id=chat_id, text='Сколько у вас детей?')
    elif last_action == 'kids':
        user_info['CNT_CHILDREN'] = int(text)
        last_action = 'birthdate'
        context.bot.send_message(chat_id=chat_id, text='Укажите дату рождения в формате ДД-ММ-ГГГГ:')
    elif last_action == 'birthdate':
        d = pd.to_datetime(text, format='%d-%m-%Y')
        r = (d.now().value - d.value) // 31536000 / (10 ** 9)
        user_info['DAYS_BIRTH'] = np.log(r)
        cats = ['Женат / Замужем', 'Холост / Не замужем', 'Гражданский брак', 'В разводе', 'Вдовец / Вдова']
        cats_call = ['married', 'single', 'civil', 'separated', 'widow']
        keyboard = []
        for c, cc in zip(cats, cats_call):
            keyboard.append([InlineKeyboardButton(c, callback_data = 'status_' + cc)])
        context.bot.send_message(chat_id=chat_id, text='Укажите ваше семейное положение:', reply_markup=InlineKeyboardMarkup(keyboard))

def get_banks_decisions(user_info):
    pred_columns = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH', 'Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
    info = []
    for col in pred_columns:
        info.append(user_info[col])
    info = np.array([info])
    train_csv = pd.read_csv('train.csv', index_col=0)
    y_train = train_csv['TARGET']
    train_csv.drop(columns=['TARGET'], inplace=True)
    stad_scaler = StandardScaler()
    X_train = stad_scaler.fit_transform(train_csv)
    lg = LogisticRegression()
    lg.fit(X_train, y_train)
    scaled_info = stad_scaler.transform(info)
    return (int(lg.predict_proba(scaled_info)[0, 1] > 0.5), int(lg.predict_proba(scaled_info)[0, 1] > 0.7))
    

def calculate_credit_value(needed_value, percentage, months = 36):
    percentage_month = percentage / 12 
    return needed_value * (percentage_month + (percentage_month / ((1 + percentage_month) ** months - 1)))

def check_end_auction(banks_min_percentage, banks_cur_percentage):
    for i, j in zip(banks_min_percentage, banks_cur_percentage):
        if j < i:
            return False
    return True

def auction_starter(user_info, banks_base_percentage, step=0.01):
    needed_value = np.exp(user_info['AMT_CREDIT']) / 8
    banks_min_percentage = [0.099, 0.12]
    banks_min_offers = [calculate_credit_value(needed_value, x) for x in banks_min_percentage]
    banks_cur_percentage = banks_base_percentage
    banks_cur_offers = [calculate_credit_value(needed_value, x) for x in banks_cur_percentage]
    best_offer = min(banks_cur_offers)
    last_call = np.argmin(banks_cur_offers)
    while check_end_auction(banks_min_percentage, banks_cur_percentage):
        for i in range(len(banks_min_offers)):
            if i != last_call:
                next_perc = min(banks_cur_percentage) - step
                banks_cur_percentage[i] = next_perc
                banks_cur_offers[i] = calculate_credit_value(needed_value, next_perc)
                last_call = i
                
    winner = np.argmax(banks_cur_percentage)
    return winner, min(banks_cur_percentage)
    
def clear(update, context):
    chat_id = update.effective_message.chat_id
    with open('test.file') as fin:
        t = fin.read()
    context.bot.send_message(chat_id=chat_id, text=t)
    user_info = dict()
    global last_action
    last_action = ""
    
def main():
    updater = Updater(TOKEN, use_context=True)

    start_handler = CommandHandler("start", send_welcome)
    clear_handler = CommandHandler("clear", clear)
    buttons_handler = CallbackQueryHandler(callback=keyboard_callback_handler)
    message_handler = MessageHandler(Filters.text, feedback)
    updater.dispatcher.add_handler(start_handler)
    updater.dispatcher.add_handler(buttons_handler)
    updater.dispatcher.add_handler(message_handler)
    updater.dispatcher.add_handler(clear_handler)

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
