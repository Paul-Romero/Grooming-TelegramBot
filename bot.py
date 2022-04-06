# Importación de librerías
# from telegram import InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import Updater, CallbackContext, CommandHandler, MessageHandler, Filters
from telegram import Update
import logging, os, joblib, emoji

# Configuración básica del login
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Obtener el token guardado en una variable de entorno por seguridad
TOKEN = os.getenv('TOKEN') #5206591761:AAGyi_D9o53DbG8Za_33kdo7j-2np5qDQWU

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
    return emoji.get_emoji_regexp().sub('ja', text)

# Función para evaluar la probabilidad de grooming del mensaje
def check_groom(text):
    niv_groom = []
    try:
        while len(niv_groom) < 5:
            msg_groom_prob = clf.predict_proba([text])[0][2]*100 # Calcula la probabilidad de que sea grooming
            niv_groom.append(msg_groom_prob) # Agrega el valor probabilístico a una lista
            groom_prom = sum(niv_groom)/len(niv_groom) # Calcula el promedio probabilístico de la lista
            if groom_prom > 90.0:
                return groom_prom # Si el promedio porbabilísitico es mayor a 90 lo devuelve
    except TypeError as te:
        raise(f"{te} \n Ocurrió un error en el tipo de dato: {type(text)}")
    return 0

# Método que genera una alerta en el chat si existe contexto grooming
def alert(update: Update, context: CallbackContext):
    msg = update.message.text # Obtiene el texto del mensaje recivido en el chat
    text = delete_emojis(msg) # Descarta los emojis del texto
    logger.info(f"El usuario {update.effective_user['username']} ha enviado un mensaje: {msg}") # Muestra un log del mensaje
    msg_groom_prob = check_groom(text) # Evalua la probabilidad de que sea un mensaje grooming
    try:
        if msg_groom_prob >= 95.0:
            logger.info(f"El chat posee {msg_groom_prob:.2f}% de grooming.")
            # Si la probabilidad es mayor a 90 emite una alerta en el chat
            context.bot.send_message(chat_id=update.effective_chat.id, text="🚫 ¡Alerta! contenido grooming en el chat.")
    except TypeError as te:
        raise(f"{te} \n Ocurrió un error en el tipo de dato: {type(msg), type(msg_groom_prob)}")

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
def run(updater):
    PORT = int(os.environ.get('PORT','443')) # Puerto que acepta la API de Telegram
    HEROKU_APP_NAME = os.environ.get("HEROKU_APP") # Acepta la variable de entorno del nombre de la app
    updater = Updater(TOKEN)
    updater.start_webhook(listen='0.0.0.0', port=PORT, url_path=TOKEN)
    updater.bot.set_webhook(f"http://{HEROKU_APP_NAME}.herokuapp.com/{TOKEN}")
    print("Bot cargado...")

run(updater)