from lightly.loss.ntx_ent_loss import NTXentLoss
from ACILoss.Experiment import Experiment
import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Calculates result for model",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-n", "--name", type=str, help="Name of the experiment.", required=True)
    parser.add_argument("-d", "--dataset", type=str, help="Dataset to use.", required=True)

    parser.add_argument("-i", "--input_size", type=int, help="The ammount of data per window used.", required=True),
    parser.add_argument("-o", "--output_size", type=int, help="The horizon of prediction required.", required=True)
    parser.add_argument("-e", "--emb_size", type=int, help="The embedding size used.", required=False, default=168)


    parser.add_argument("-s", "--seed", type=int, help="The seed for replication.", required=False, default=1812)
    parser.add_argument("-m", "--mode", type=str, help="Method employed for the embedding", required=False, default=None)
    parser.add_argument("-b", "--batch_size", type=int, help="The number of observations per batch", required=False, default=32)

    parser.add_argument("-l", "--scaler", type=float, help="The weighting scale for the Loss.", required=False, default=.25)
    parser.add_argument("-N", "--epochs", type=int, help="The number of iterations for training.", required=False, default=200)
    parser.add_argument("-P", "--patience", type=int, help="Number of iterations without improving before early stopping.", required=False, default=30)
    parser.add_argument("-p", "--scheduler_patience", type=int, help="Number of iterations without improving before dropping learning rate.", required=False, default=10)
    parser.add_argument("-w", "--weight_decay", type=float, help="The weighting decay for Adam algorithm.", required=False, default=1e-4)
    args = parser.parse_args()

    ex = Experiment(name=args.name, 
                    dataset=args.dataset,
                    input_size=args.input_size,
                    emb_size = args.emb_size,
                    output_size = args.output_size,
                    mode = args.mode if args.mode != "None" else "",
                    seed = args.seed,
                    batch_size= args.batch_size)

    ex.fit(scaler=args.scaler,
           epochs=args.epochs,
           patience=args.patience,
           scheduler_patience=args.scheduler_patience,
           weight_decay = args.weight_decay)
    
if __name__ == "__main__":
    main()
