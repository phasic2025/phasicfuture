# Command Interface Guide

## 🎮 Interactive Commands

The neural network now has a command interface! Send commands to control and monitor the network.

## Available Commands

### Control Commands

**`speed up`** or **`faster`**
- Advances simulation time
- Use to accelerate progress

**`add neurons`** or **`add`**
- Adds 10 more neurons to the network
- Increases network capacity
- Example: "add neurons" → Network grows from 50 to 60 neurons

**`reset`** or **`restart`**
- Resets entire network
- Clears all progress
- Starts fresh toaster design from beginning

### Status Commands

**`status`**
- Shows current state:
  - Goals achieved
  - Synchronization level
  - Simulation time
- Example output: "Status: 2/5 goals achieved, Sync: 0.756, Time: 12.34s"

**`help`**
- Lists all available commands

## How to Use

1. **Open**: http://localhost:8080
2. **Find**: Command Interface panel at bottom
3. **Type**: Your command in the input box
4. **Send**: Click "Send" or press Enter
5. **Watch**: Response appears in output area

## Example Session

```
> status
[12:34:56] Status: 1/5 goals achieved, Sync: 0.623, Time: 8.45s

> speed up
[12:34:57] Sped up simulation

> add neurons
[12:34:58] Added neurons. Total: 60

> reset
[12:35:00] Network reset. Starting fresh toaster design.
```

## What's Fixed

✅ **Time steps now updating** - Fixed data fetching with cache busting
✅ **Command interface** - Send commands to control the network
✅ **Real-time updates** - Data refreshes every 100ms
✅ **Debug logging** - Check browser console (F12) to see updates

## Troubleshooting

**Time not updating?**
- Open browser console (F12)
- Check for errors
- Look for "Update #X, Time: Y" messages

**Commands not working?**
- Make sure server is running
- Check browser console for errors
- Try "help" command first

**Network not visible?**
- Refresh page (F5)
- Check server logs: `tail -f /tmp/neural_server.log`

---

**Ready to command!** Type your first command and watch the network respond! 🎮

