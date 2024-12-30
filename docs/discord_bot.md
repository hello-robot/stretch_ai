 Set up a Discord Bot for Stretch

*WARNING: following this document will let you connect your robot to a website, [Discord](https://discord.com/), that is not under your control. Be very careful with this, and make sure you understand the risks involved.* The robot will only be connected while you are running a specific script.

To create a discord bot to chat with your stretch, you'll need to use the [Discord Developer Portal](https://discord.com/developers/applications) to create a custom application. This document will walk you through the steps to authorize the app in discord and to get the token needed to run the bot. These instructions are based on the [Discord Developer Portal Documentation](https://discord.com/developers/docs/intro) and on [a previous writeup](https://github.com/cpaxton/virgil/blob/main/docs/discord.md).

On your laptop, in your python environment, you will need to install the discord package:

```bash
python -m pip install discord.py python-dotenv
```

![Create a New server](images/discord0.png)

First, if you have not done this, you need to create a Discord server, create a [Discord](https://discord.com/) account and login into it, create a server (with a name you like) and a channel named as `#talk-to-stretch`.

![Create a New Application](images/discord_create_app.png)

Next, you will go into the developer portal and click on the "New Application" button. This will allow you to create a new application that will be your bot.

![Name your Application](images/discord_name_app.png)

Enter a name. We suggest naming your new bot after your Stretch, since you will be using it to chat with your robot specifically. In the example above, I named my bot "Stretch3005."

After this, I can see Stretch3005 listed under my applications. Click on it to continue.

## Get installation link

![Installation Tab](images/discord_install_page.png)

You will need to retrieve the installation link for your bot from the Installation tab.
```
https://discord.com/oauth2/authorize?client_id=$CLIENT_ID
```

where `$CLIENT_ID` is the client ID of your bot.

## Get OAuth2 Token

![Oath2 redirects](images/discord_oauth2_redirects.png)

Copy and paste the installation link under Redirects.

![Discord Bot Permissions](images/discord_bot_permissions.png)

Then make sure the bot has the correct permissions. You'll need `guild`, `messages.read`, and `bot` permissions at a minimum. Then, choose a redirect URL (the one you just entered above), and add the correct Bot permissions as well.

The bot permissions need to include sending files and messages.

Then you can create an install link. This will be the link you use to add the bot to your server. Integration type should be "guild install" -- guild is the internal term for a server in the Discord API.

![Discord Confirmation](images/discord_confirmation.png)

Finally, you'll get a URL you can copy into Discord and use to install your bot in a server. Copy and paste it into Discord (in a normal chat is fine), and click on the link to add the bot to your server.

## Set Privileged Intents and Get Token

![Bot Token](images/discord_bot_token.png)

Get the token from the Bot tab. If this is the first time you used this discord app, you can just click `Reset Token`

Then set privileged intents to let it send messages, send images, and join the server.

![Privileged Intents](images/discord_get_intents.png)

## Set the token in the command line

The discord bot will read the token from the command line:

```bash
export DISCORD_TOKEN=$TOKEN
```

where `$TOKEN` is the one you got [here](#set-privileged-intents-and-get-token). You can find this on the Installation page or in the OAuth2 Client Information section.

## Running the bot

You can now run the bot with the following command:

```bash
python3 -m stretch.app.run_discord --robot_ip $ROBOT_IP --llm openai
```

*BE VERY CAREFUL WITH THIS.* This command can and will send images from your robot to the internet! Know your environment, and make sure that you are using the robot in a safe and secure location. Do not leave the robot unnatended while running this command.

Chat with the bot via the `#talk-to-stretch` channel in the discord server. You can ask it to take pictures, move, and more.

| ![Discord Chat](images/discord1.png) | ![Discord Chat](images/discord2.png) |
|--------------|------------------|

For more information on the specific usage of the script, you can see the [LLM Agent docs](llm_agent.md) and the [DynaMem docs](dynamem.md).
