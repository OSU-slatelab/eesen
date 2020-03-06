from utils.fileutils import debug
import os, sys
import constants

def check_sets_training(tr_data, cv_data):
    """ Basic data consistency checks """

    # Check for same data types
    for key in tr_data:
        if key not in cv_data:
            raise ValueError("Train and CV should have same structure")

    # Check for same languages
    for language in tr_data['x'].get_language_scheme():
        for key in tr_data:
            if language not in tr_data[key].get_language_scheme() or \
                    language not in cv_data[key].get_language_scheme():
                raise ValueError("Language '{}' not consistent".format(language))

    for language, targets_dic in tr_data['y'].get_language_scheme().items():
        for target_id, number_targets in targets_dic.items():
            if target_id not in cv_data['y'].get_language_scheme()[language]:
                print("Error: target: "+target_id+" not find in tr_data['y']\n")
                print(debug.get_debug_info())
                print("exiting... \n")
                sys.exit()

            if number_targets != cv_data['y'].get_language_scheme()[language][target_id]:
                print(80 * "*")
                print(80 * "*")
                print("WARINING!: number of targets ("+str(number_targets)+") from tr_data['y'] ("+str(cv_data['y'].get_language_scheme()[language][target_id])+") in language: "+str(language)+" in target: "+str(target_id)+"is different form cv_data['y']")
                print(debug.get_debug_info())
                print("replicating biggest one...")
                if(number_targets > cv_data['y'].get_language_scheme()[language][target_id]):
                    cv_data['y'].set_number_targets(language, target_id, number_targets)
                else:
                    tr_data['y'].set_number_targets(language, target_id, cv_data['y'].get_language_scheme()[language][target_id])
                print(80 * "*")
                print(80 * "*")

        print("languages checked ...")
        print("(cv_data['x'] vs cv_data['y'] vs tr_data['x'] vs tr_data['y'])")

def check_sets_testing(config, test_x, sat_x = None):
    if config[constants.INPUT_FEATS_DIM] != test_x.get_num_dim():
        print("Error: input dimension from model loaded({}) is not the same as input_feats ({})".format(
            config[constants.INPUT_FEATS_DIM],
            test_x.get_num_dim()),
        )
        print(debug.get_debug_info())
        print("exiting... \n")
        sys.exit()

    if sat_x and config[constants.SAT_FEAT_DIM] != sat_x.get_num_dim():
        print("Error: input sat dimension from model loaded({}) is not the same as input_feats ({})".format(
            config[constants.SAT_FEAT_DIM],
            sat_x.get_num_dim()),
        )
        print(debug.get_debug_info())
        print("exiting... \n")
        sys.exit()

def check_sat_exist(config, tr_x):

    language_general_flag = False
    for language in tr_x.get_language_scheme():

        if len(tr_x.get_language_scheme()) > 1:
            lan_path = os.path.join(config[constants.CONF_TAGS.DATA_DIR], language)
        else:
            lan_path = config[constants.CONF_TAGS.DATA_DIR]

        language_flag = False
        for lan_file in os.listdir(lan_path):
            if os.path.splitext(lan_file)[0] == constants.DEFAULT_FILENAMES.SAT:
                print("speaker adaptation vector found for language: "+language+"\n")
                language_flag = True
                language_general_flag = True

        if language_general_flag:
            if not language_flag:
                print("Error: inconsisten structure over language on sat (local_sat.scp) files\n")
                print(debug.get_debug_info())
                print("exiting... \n")
                sys.exit()
                return False
            else:
                return True
