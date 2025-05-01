import yaml
import logging
import argparse
from module import Preprocessor, Trainer, Predictor

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process commandline')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_level', type=str, default="INFO")
    args = parser.parse_args()

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level = args.log_level)
    logger = logging.getLogger('global_logger')

    logger.info("Start!")

    with open(args.config, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            logger.info('run.py line 21')
            preprocessor = Preprocessor(config['preprocessing'], logger)
            logger.info('run.py line 23')
            _, _, train_x, train_y, validate_x, validate_y, test_x = preprocessor.process()
            logger.info('run.py line 25')
            if config['training']['model_name'] != 'naivebayes':
                config['training']['vocab_size'] = len(preprocessor.word2ind.keys())
            logger.info('run.py line 28')

            trainer = Trainer(config['training'], logger, preprocessor.classes)
            model = trainer.fit(train_x, train_y)
            accuracy, cls_report = trainer.validate(validate_x, validate_y)
            logger.info("accuracy:{}".format(accuracy))
            logger.info("\n{}\n".format(cls_report))
            predictor = Predictor(config['predict'], logger, model)
            probs = predictor.predict_prob(test_x)
            predictor.save_result(preprocessor.test_ids, probs)
        except yaml.YAMLError as err:
            logger.warning("Config file error: {}".format(err))

    logger.info("Completed!")