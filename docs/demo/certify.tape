# Records docs/demo/certify.gif from a real run: `vhs docs/demo/certify.tape` from the repo root.
Output docs/demo/certify.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 480
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' RUNS=../.bench-data/adversarial-robustness-lab/runs && source .venv/bin/activate && clear` Enter
Sleep 2s
Show
Type `python -m adversarial_robustness_lab certify -c $RUNS/wrn16_2_noise025.pt --n 8 --n-samples 1000 --device cpu` Enter
Wait+Line@300s /lab\) \$/
Sleep 5s
