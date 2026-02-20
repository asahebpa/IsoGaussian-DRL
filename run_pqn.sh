#!/bin/bash

# Define 10 environments
OUTPUT_DIR="./slurm_logs/pqn"
mkdir -p "$OUTPUT_DIR"


environments=(
    "Alien-v5" "Amidar-v5" "Asteroids-v5" "Assault-v5"
    "BankHeist-v5" "BattleZone-v5" "BeamRider-v5" "Berzerk-v5"
    "Bowling-v5" "Boxing-v5" "Breakout-v5" "Centipede-v5" "CrazyClimber-v5"
    "Defender-v5" "DemonAttack-v5" "DoubleDunk-v5" 
    "Enduro-v5" "Freeway-v5" "Frostbite-v5"
    "Gopher-v5" "Gravitar-v5" "Hero-v5" "IceHockey-v5" "Jamesbond-v5"
    "Kangaroo-v5" "Krull-v5" "KungFuMaster-v5" "MontezumaRevenge-v5"
    "MsPacman-v5" "NameThisGame-v5" "Pitfall-v5" "PrivateEye-v5"
    "Riverraid-v5" "RoadRunner-v5" "Robotank-v5" "Seaquest-v5" "Skiing-v5"
    "Solaris-v5" "SpaceInvaders-v5" "StarGunner-v5" "Surround-v5" "Tennis-v5" "TimePilot-v5"
    "Tutankham-v5" "UpNDown-v5" "Venture-v5" "VideoPinball-v5" "WizardOfWor-v5"
    "YarsRevenge-v5" "Zaxxon-v5" "ChopperCommand-v5" "Asterix-v5" 
    "Atlantis-v5" "FishingDerby-v5" "Pong-v5" "Qbert-v5" "Phoenix-v5" 
)


schemes=(
    "radam_sigreg0:radam:0.0"
    "radam_sigregp2:radam:0.2"
    "radam_sigreg1:radam:1.0"
)


seeds=(1 2 3 4)

# Submit jobs for each combination
for env in "${environments[@]}"; do
    for scheme in "${schemes[@]}"; do
        IFS=':' read -r suffix opt sigreg_lambda <<< "$scheme"

        for seed in "${seeds[@]}"; do
            job_name="PQN_${env}_${suffix}_seed${seed}"
            output_file="${OUTPUT_DIR}/${job_name}.txt"

            # Build command with optimizer and sigreg_lambda
            cmd="python ./pqn.py --mlp_type="default"  --mlp_depth="medium" --env_id=$env --seed=$seed --optimizer=$opt --sigreg_lambda=$sigreg_lambda"

            # Submit job
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --output=$output_file
#SBATCH --error=$output_file
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=20Gb
#SBATCH --gres=gpu:1
#SBATCH -c 32

module load anaconda/3
conda activate deeprl
$cmd
EOF

            echo "Submitted: $job_name"
            sleep 0.3
        done
    done
done

echo "All jobs submitted!"
echo "Total jobs: $((${#environments[@]} * ${#schemes[@]} * ${#seeds[@]}))"