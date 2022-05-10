# Importación de librerías
# from telegram import InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import Updater, CallbackContext, CommandHandler, MessageHandler, Filters
from telegram import Update
from time import localtime, strftime
from itertools import islice
import logging, os, joblib, emoji

# Configuración básica del login
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Obtener el token guardado en una variable de entorno por seguridad
TOKEN = os.getenv('TOKEN')

# Enlazar el Token con el Updater de la API de Telegram
updater = Updater(token=TOKEN, use_context=True)

# Declarar el controlador del Bot
dispatcher = updater.dispatcher

# Importar el clasificador
clf = joblib.load('GroomerClassifier.pkl')

# Método que muestra un saludo de inicio del Bot
def start(update: Update, context: CallbackContext):
    logger.info(f"El usuario {update.effective_user['username']} ha iniciado el chat.")
    context.bot.send_message(chat_id=update.effective_chat.id, text="Hello world! 🤖")

# Método para comandos desconocidos
def unknown(update: Update, context: CallbackContext):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Lo siento no comprendo tu comando.")

# Función para descartar emojis del texto
def delete_emojis(text):
    return emoji.replace_emoji(text, replace='ja')

# Método para registrar la información
def logs(data):
    try:
        time_now = strftime('%B %d %Y %H:%M:%S', localtime()) # Obtiene la hora local
        with open('./logs.txt', 'a+') as file_logs:
            file_logs.write(time_now + '\t' + str(data['text']) + '\t' + str(data['niv_groom']) + '\n') # Registra la hora con el mensaje y el respectivo porcentaje
    except FileNotFoundError:
        raise('No se encontró el archivo')

# Función para evaluar la probabilidad de grooming del mensaje
def check_groom(text):
    niv_groom = clf.predict_proba([text])[0][2]*100
    return niv_groom # Devuelve un porcentaje de grooming del mensaje

    """niv_groom = []
    try:
        while len(niv_groom) < 5:
            msg_groom_prob = clf.predict_proba([text])[0][2]*100 # Calcula la probabilidad de que sea grooming
            niv_groom.append(msg_groom_prob) # Agrega el valor probabilístico a una lista
            groom_prom = sum(niv_groom)/len(niv_groom) # Calcula el promedio probabilístico de la lista
            if groom_prom > 90.0:
                return groom_prom # Si el promedio porbabilísitico es mayor a 90 lo devuelve
    except TypeError as te:
        raise(f"{te} \n Ocurrió un error en el tipo de dato: {type(text)}")
    return 0"""

# Método que genera una alerta en el chat si existe contexto grooming
def alert(update: Update, context: CallbackContext):
    cnt = 0
    msg = update.message.text # Obtiene el texto del mensaje recivido en el chat
    text = delete_emojis(msg) # Descarta los emojis del texto
    niv_groom = check_groom(text) # Evalua la probabilidad de que sea un mensaje grooming
    data = {'text': text, 'niv_groom': niv_groom} # Almacena en la variable el texto con su nivel de grooming
    logger.info(f"El usuario {update.effective_user['username']} ha enviado un mensaje: {msg} con {niv_groom:.2f}% grooming") # Muestra un log del mensaje
    logs(data) # Registra la información en un archivo externo
    try:
        if msg: cnt += 1
        list_niv_groom = [] # Lista para los porcentajes de grooming de cada mensaje
        with open('./logs.txt', 'r') as file_logs: # Abrir el registro de mensajes
            for item in islice(file_logs, cnt): # Lee los 5 primeros mensajes del registro
                value = float(item.split('\t')[2]) # Toma los valores porcentuales de cada mensaje
                list_niv_groom.append(value) # Agrega el porcentaje de grooming a la lista para promediar
        msg_groom_prob = sum(list_niv_groom)/len(list_niv_groom) # Promedio probabilistico de grooming en el chat
        print(f'Lista de porcentajes: {list_niv_groom} \n Promedio total: {msg_groom_prob}')
        if msg_groom_prob > 80.0 and len(list_niv_groom) == cnt: # Si la probabilidad es mayor a 80 emite una alerta en el chat
            logger.info(f"El chat posee {msg_groom_prob:.2f}% de grooming.") # Registro en consola
            context.bot.send_message(chat_id=update.effective_chat.id, text="🚫 ¡Alerta! contenido grooming en el chat.")
    except TypeError as te:
        raise(f"{te} \n Ocurrió un error en el tipo de dato: {type(msg), type(niv_groom)}")
    except FileNotFoundError:
        raise("No se encontró el archivo")

# Pasar el método de inicio al controlador para la interacción con el usuario
start_handler = CommandHandler('start', start)
dispatcher.add_handler(start_handler)

# Agregar el método para comandos desconocidos
unknown_handler = MessageHandler(Filters.command, unknown)
dispatcher.add_handler(unknown_handler)

# Agregar el método para generar la alert
groom_handler = MessageHandler(Filters.text, alert)
dispatcher.add_handler(groom_handler)

# Comenzar la ejecución del Bot
def run_webhook(updater):
    PORT = int(os.environ.get('PORT','8443')) # Puerto que acepta la API de Telegram
    # HEROKU_APP_NAME = os.environ.get("HEROKU_APP") # Acepta la variable de entorno del nombre de la app
    updater.start_webhook(listen='0.0.0.0', port=PORT, url_path=TOKEN, webhook_url="https://grooming-telegram-bot.herokuapp.com/"+TOKEN) # inicia el webhook de la app en heroku
    updater.idle() # Finaliza el bot con ctrl+c

#updater.start_polling()
run_webhook(updater)