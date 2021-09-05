import run_Encoder
import run_VAE
# from run_VAE import run_VAE


def main():
    run_VAE.run_VAE()
    run_Encoder.run_encoder()


if __name__ == '__main__':
    main()
