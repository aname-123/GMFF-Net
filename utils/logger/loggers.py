import csv
import os.path


def write_to_csv(save_dir,
                 epoch,
                 train_loss,
                 lr,
                 val_loss,
                 avg_error):
    # path to save result.csv
    csv_file = os.path.join(save_dir, "results.csv")

    # Columns name for the CSV file
    fieldnames = ['epoch', 'train_loss', 'lr', 'val_loss', 'avg_error']
    # Write to CSV file
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header if file is empty
        if f.tell() == 0:
            writer.writeheader()

        # Write data for the current epoch
        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': lr,
            'val_loss': val_loss,
            'avg_error': avg_error
        })


if __name__ == "__main__":
    save_dir = "./"
    epoch = 0
    train_loss = 0
    lr = 0.001
    val_loss = 0
    avg_error = 11.1
    write_to_csv(save_dir, epoch, train_loss, lr, val_loss, avg_error)
