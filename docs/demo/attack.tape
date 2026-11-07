# Records docs/demo/attack.gif from a real run: `vhs docs/demo/attack.tape` from the repo root.
Output docs/demo/attack.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 400
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' RUNS=../.bench-data/adversarial-robustness-lab/runs && source .venv/bin/activate && clear` Enter
Sleep 2s
Show
Type `python -m adversarial_robustness_lab attack -c $RUNS/wrn16_2_standard.pt --attack pgd --n 6` Enter
Sleep 12s
Type `python -m adversarial_robustness_lab attack -c $RUNS/wrn16_2_pgd.pt --attack pgd --n 6` Enter
Sleep 12s
