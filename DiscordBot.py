import discord
from mimetypes import guess_extension

# keeps track of all files waiting in the queue. generally there will only be one of these, assuming the bot works quickly enough
class QueueWaiter():
    def __init__(self, message, file):
        self.message = message
        self.file = file

# inherits from the discord.Client class and handles messages it receives
class VoiceAI(discord.Client):
    def __init__(self, converter, density_threshold, neural, tester, output_folder, **options):
        super().__init__(**options)
        self.converter = converter
        self.density = density_threshold
        self.neural = neural
        self.tester = tester
        self.output = output_folder
        self.running = False
        self.queue = []

    # runs after the bot finishes initializing
    async def on_ready(self):
        print("VoiceAI is now online\nAll @ mentions and DMs to the bot with an audio clip will return its output image.")

    # runs when the bot receives a message, whether it's in a server it's active in or if it's in a DM
    async def on_message(self, message):
        if message.author == self.user:
            return
        if message.author.bot is True:
            return
        if message.guild is None or self.user in message.mentions:
            print("Received message from " + str(message.author))
            audio_file = None
            if message.attachments:
                for attachment in message.attachments:
                    if "audio" in str(attachment.content_type):
                        audio_file = attachment
                        break
            if not audio_file is None:
                print("Received audio file from " + str(message.author))
                self.queue.append(QueueWaiter(message, audio_file))
                await self.process_queue()
            else:
                if message.author.id == 177971643761557504:
                    if "!shutdown" in message.content:
                        await self.close()
                else:
                    await message.reply("You need to attach an audio clip to your message!")

    # processes the queue, and prevents itself from being run multiple times asynchronously
    async def process_queue(self):
        if self.running:
            print("Already running process_queue")
            return
        self.running = True
        print("Processing queue")
        while len(self.queue) > 0:
            waiter = self.queue.pop(0)
            try:
                save_path = "input\\single_clip\\clip" + guess_extension(str(waiter.file.content_type))
            except:
                await waiter.message.reply("Could not process clip: File is not an acceptable data type. Please try a different format.")
                continue
            await waiter.file.save(save_path)
            result = self.tester.run_single(save_path, self.converter, self.neural, self.output, self.density)
            if result is None:
                await waiter.message.reply(file=discord.File(self.output + "\\prediction_image\\prediction.jpg"))
            else:
                await waiter.message.reply("Could not process clip: " + result)
        self.running = False