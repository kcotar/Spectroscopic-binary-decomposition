from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from shutil import copyfile
from astropy.table import Table
from pyraf import iraf
from glob import glob

def export_observation_to_txt(fits_path, txt_path):
	print(' Exporting file')
	for order in np.arange(1, 32, 1):
		try:
			iraf.wspectext(input=fits_path+'[*,'+str(order)+',1]', output=txt_path+'_{:.0f}.txt'.format(order), header='no')
		except Exception as e:
			print(e)
			pass


iraf.noao(_doprint=0, Stdout="/dev/null")
iraf.rv(_doprint=0, Stdout="/dev/null")
iraf.imred(_doprint=0, Stdout="/dev/null")
iraf.ccdred(_doprint=0, Stdout="/dev/null")
iraf.images(_doprint=0, Stdout="/dev/null")
iraf.immatch(_doprint=0, Stdout="/dev/null")
iraf.onedspec(_doprint=0, Stdout="/dev/null")
iraf.twodspec(_doprint=0, Stdout="/dev/null")
iraf.apextract(_doprint=0, Stdout="/dev/null")
iraf.imutil(_doprint=0, Stdout="/dev/null")
iraf.echelle(_doprint=0, Stdout="/dev/null")
iraf.astutil(_doprint=0, Stdout="/dev/null")
iraf.apextract.dispaxi = 1
iraf.echelle.dispaxi = 1
iraf.ccdred.instrum = 'blank.txt'
os.environ['PYRAF_BETA_STATUS'] = '1'
os.system('mkdir uparm')
iraf.set(uparm=os.getcwd() + '/uparm')

data_dir = '/data4/travegre/Projects/Asiago_binaries/'
reduc_dir = '/data4/travegre/Projects/Echelle_Asiago_Reduction/delo/observations/'

observations = {  

  'binaries_13_201502':  {'calibs': ['EC56031', 'EC56031', 'EC56033', 'EC56036', 'EC56036', 'EC56038'], 'REF_AP': '', 'objects': ['EC56030', 'EC56032', 'EC56034', 'EC56035', 'EC56037', 'EC56039'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201502/', 'ap_position': 'left', 'biases': ['EC56002', 'EC56003', 'EC56004', 'EC56005', 'EC56006'], 'flats': ['EC56007', 'EC56008', 'EC56009', 'EC56010', 'EC56011']} ,

  'binaries_13_201504':  {'calibs': ['EC56260', 'EC56262'], 'REF_AP': '', 'objects': ['EC56261', 'EC56263'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201504/', 'ap_position': 'left', 'biases': ['EC56331', 'EC56332', 'EC56333', 'EC56334', 'EC56335'], 'flats': ['EC56322', 'EC56326', 'EC56327', 'EC56328', 'EC56329', 'EC56330']} ,

  #'binaries_13_201507':  {'calibs': [], 'REF_AP': '', 'objects': [], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201507/', 'ap_position': 'left', 'biases': ['EC56519', 'EC56520', 'EC56521', 'EC56522', 'EC56523'], 'flats': ['EC56524', 'EC56525', 'EC56526', 'EC56527', 'EC56528']} ,

  'binaries_13_201509':  {'calibs': ['EC56697', 'EC56699', 'EC56699'], 'REF_AP': '', 'objects': ['EC56696', 'EC56698', 'EC56700'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201509/', 'ap_position': 'left', 'biases': ['EC56331', 'EC56332', 'EC56333', 'EC56334', 'EC56335'], 'flats': ['EC56322', 'EC56326', 'EC56327', 'EC56328', 'EC56329', 'EC56330']} ,

  'binaries_13_201510':  {'calibs': ['EC56869', 'EC56869', 'EC56933'], 'REF_AP': '', 'objects': ['EC56868', 'EC56870', 'EC56932'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201510/', 'ap_position': 'left', 'biases': ['EC56796', 'EC56797', 'EC56798', 'EC56799', 'EC56800', 'EC56820', 'EC56821', 'EC56822', 'EC56823', 'EC56824', 'EC56828', 'EC56829', 'EC56830', 'EC56831', 'EC56832', 'EC56882', 'EC56883', 'EC56884', 'EC56885', 'EC56886'], 'flats': ['EC56801', 'EC56802', 'EC56803', 'EC56804', 'EC56805', 'EC56825', 'EC56826', 'EC56827', 'EC56876', 'EC56877', 'EC56878', 'EC56879', 'EC56880', 'EC56881']} ,

  'binaries_13_201601':  {'calibs': ['EC57450', 'EC57476', 'EC57476', 'EC57510', 'EC57510', 'EC57513', 'EC57515', 'EC57515', 'EC57518', 'EC57520', 'EC57522', 'EC57522', 'EC57525', 'EC57556', 'EC57556', 'EC57559', 'EC57566', 'EC57566', 'EC57569', 'EC57571', 'EC57571', 'EC57574', 'EC57602', 'EC57602', 'EC57605', 'EC57607', 'EC57607', 'EC57610', 'EC57612', 'EC57612', 'EC57615', 'EC57617', 'EC57617', 'EC57620'], 'REF_AP': '', 'objects': ['EC57449', 'EC57475', 'EC57477', 'EC57509', 'EC57511', 'EC57512', 'EC57514', 'EC57516', 'EC57517', 'EC57519', 'EC57521', 'EC57523', 'EC57524', 'EC57555', 'EC57557', 'EC57558', 'EC57565', 'EC57567', 'EC57568', 'EC57570', 'EC57572', 'EC57573', 'EC57601', 'EC57603', 'EC57604', 'EC57606', 'EC57608', 'EC57609', 'EC57611', 'EC57613', 'EC57614', 'EC57616', 'EC57618', 'EC57619'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201601/', 'ap_position': 'left', 'biases': ['EC57432', 'EC57433', 'EC57434', 'EC57435', 'EC57436'], 'flats': ['EC57453', 'EC57454', 'EC57455', 'EC57493', 'EC57494', 'EC57495']} ,

  #'binaries_13_201602':  {'calibs': [], 'REF_AP': '', 'objects': [], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201602/', 'ap_position': 'left', 'biases': ['EC57771', 'EC57772', 'EC57773', 'EC57774'], 'flats': ['EC57775', 'EC57776', 'EC57777']} ,

  'binaries_13_201603':  {'calibs': ['EC57806', 'EC57817', 'EC57817', 'EC57822', 'EC57825', 'EC57825', 'EC57827', 'EC57849', 'EC57849', 'EC57851', 'EC57857', 'EC57857', 'EC57859', 'EC57863', 'EC57863', 'EC57865'], 'REF_AP': '', 'objects': ['EC57805', 'EC57816', 'EC57818', 'EC57823', 'EC57824', 'EC57826', 'EC57828', 'EC57848', 'EC57850', 'EC57852', 'EC57856', 'EC57858', 'EC57860', 'EC57862', 'EC57864', 'EC57866'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201603/', 'ap_position': 'left', 'biases': ['EC57799', 'EC57800', 'EC57801', 'EC57802', 'EC57803'], 'flats': ['EC57836', 'EC57837', 'EC57838']} ,

  'binaries_13_201604':  {'calibs': ['EC57938', 'EC57929', 'EC57929', 'EC57931', 'EC57934', 'EC57934', 'EC57938'], 'REF_AP': '', 'objects': ['EC57939', 'EC57928', 'EC57930', 'EC57932', 'EC57933', 'EC57935', 'EC57937'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201604/', 'ap_position': 'left', 'biases': ['EC57942', 'EC57943', 'EC57944', 'EC57945', 'EC57946'], 'flats': ['EC57947', 'EC57948', 'EC57949']} ,

  'binaries_13_201607':  {'calibs': ['EC58021', 'EC58019', 'EC58019', 'EC58024', 'EC58024', 'EC58059', 'EC58060', 'EC58062', 'EC58082', 'EC58083', 'EC58085', 'EC58090'], 'REF_AP': '', 'objects': ['EC58022', 'EC58018', 'EC58020', 'EC58023', 'EC58025', 'EC58058', 'EC58061', 'EC58063', 'EC58081', 'EC58084', 'EC58086', 'EC58089'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201607/', 'ap_position': 'left', 'biases': ['EC58091', 'EC58092', 'EC58093', 'EC58094', 'EC58095'], 'flats': ['EC58096', 'EC58097', 'EC58098']} ,

  'binaries_13_201609':  {'calibs': ['EC58247', 'EC58247', 'EC58250', 'EC58251', 'EC58253', 'EC58256', 'EC58256', 'EC58258', 'EC58271', 'EC58271', 'EC58273', 'EC58276', 'EC58276', 'EC58278', 'EC58281', 'EC58281', 'EC58283', 'EC58285', 'EC58288', 'EC58288', 'EC58290', 'EC58293', 'EC58293', 'EC58295', 'EC58309', 'EC58309', 'EC58313', 'EC58313', 'EC58315', 'EC58318', 'EC58318', 'EC58330', 'EC58330', 'EC58332', 'EC58335'], 'REF_AP': '', 'objects': ['EC58246', 'EC58248', 'EC58249', 'EC58252', 'EC58254', 'EC58255', 'EC58257', 'EC58259', 'EC58270', 'EC58272', 'EC58274', 'EC58275', 'EC58277', 'EC58279', 'EC58280', 'EC58282', 'EC58284', 'EC58286', 'EC58287', 'EC58289', 'EC58291', 'EC58292', 'EC58294', 'EC58296', 'EC58308', 'EC58310', 'EC58312', 'EC58314', 'EC58316', 'EC58317', 'EC58319', 'EC58329', 'EC58331', 'EC58333', 'EC58334'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201609/', 'ap_position': 'left', 'biases': ['EC58265', 'EC58266', 'EC58267', 'EC58268', 'EC58269'], 'flats': ['EC58304', 'EC58305', 'EC58306'], 'JOIN':[['EC58275', 'EC58277', 'EC58279']]} ,

  'binaries_13_201610':  {'calibs': ['EC58432', 'EC58435', 'EC58435', 'EC58419', 'EC58417', 'EC58417', 'EC58430', 'EC58430', 'EC58437'], 'REF_AP': '', 'objects': ['EC58433', 'EC58434', 'EC58436', 'EC58420', 'EC58416', 'EC58418', 'EC58429', 'EC58431', 'EC58438'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201610/', 'ap_position': 'left', 'biases': ['EC58442', 'EC58443', 'EC58444', 'EC58445', 'EC58446'], 'flats': ['EC58447', 'EC58448', 'EC58449', 'EC58450']} ,

  'binaries_13_201612':  {'calibs': ['EC58686', 'EC58686', 'EC58688', 'EC58691', 'EC58691', 'EC58693', 'EC58696', 'EC58696', 'EC58698', 'EC58703', 'EC58703', 'EC58705', 'EC58708', 'EC58708', 'EC58716', 'EC58716', 'EC58718', 'EC58726', 'EC58726', 'EC58728', 'EC58731', 'EC58731', 'EC58733', 'EC58736', 'EC58736', 'EC58738', 'EC58741', 'EC58741', 'EC58743', 'EC58746', 'EC58747', 'EC58749', 'EC58770', 'EC58770', 'EC58772', 'EC58775', 'EC58777', 'EC58782', 'EC58782', 'EC58784', 'EC58787', 'EC58787', 'EC58789', 'EC58792', 'EC58812', 'EC58812', 'EC58814', 'EC58817', 'EC58817', 'EC58819'], 'REF_AP': '', 'objects': ['EC58685', 'EC58687', 'EC58689', 'EC58690', 'EC58692', 'EC58694', 'EC58695', 'EC58697', 'EC58699', 'EC58702', 'EC58704', 'EC58706', 'EC58707', 'EC58709', 'EC58715', 'EC58717', 'EC58719', 'EC58725', 'EC58727', 'EC58729', 'EC58730', 'EC58732', 'EC58734', 'EC58735', 'EC58737', 'EC58739', 'EC58740', 'EC58742', 'EC58744', 'EC58745', 'EC58748', 'EC58750', 'EC58769', 'EC58771', 'EC58773', 'EC58774', 'EC58778', 'EC58781', 'EC58783', 'EC58785', 'EC58786', 'EC58788', 'EC58790', 'EC58791', 'EC58811', 'EC58813', 'EC58815', 'EC58816', 'EC58818', 'EC58820'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201612/', 'ap_position': 'left', 'biases': ['EC58806', 'EC58807', 'EC58808', 'EC58809', 'EC58810'], 'flats': ['EC58803', 'EC58804', 'EC58805', 'EC58932', 'EC58933']} ,

  'binaries_13_201701':  {'calibs': ['EC58944', 'EC58944', 'EC58946', 'EC58950', 'EC58950', 'EC58952', 'EC58955', 'EC58955', 'EC58957', 'EC58960', 'EC58961', 'EC58963', 'EC58966', 'EC58966', 'EC58968', 'EC58971', 'EC58971', 'EC58973', 'EC58976', 'EC58978', 'EC58978', 'EC58980', 'EC58983', 'EC58983', 'EC58985', 'EC58988', 'EC58988', 'EC58990', 'EC58995', 'EC58995', 'EC58997', 'EC59007', 'EC59007', 'EC59009', 'EC59012', 'EC59012', 'EC59014', 'EC59019', 'EC59019', 'EC59021'], 'REF_AP': '', 'objects': ['EC58943', 'EC58945', 'EC58947', 'EC58949', 'EC58951', 'EC58953', 'EC58954', 'EC58956', 'EC58958', 'EC58959', 'EC58962', 'EC58964', 'EC58965', 'EC58967', 'EC58969', 'EC58970', 'EC58972', 'EC58974', 'EC58975', 'EC58977', 'EC58979', 'EC58981', 'EC58982', 'EC58984', 'EC58986', 'EC58987', 'EC58989', 'EC58991', 'EC58994', 'EC58996', 'EC58998', 'EC59006', 'EC59008', 'EC59010', 'EC59011', 'EC59013', 'EC59015', 'EC59018', 'EC59020', 'EC59022'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201701/', 'ap_position': 'left', 'biases': ['EC58935', 'EC58936', 'EC58937', 'EC58938', 'EC58939'], 'flats': ['EC58934', 'EC58940', 'EC58941', 'EC58942']} ,

  'binaries_13_201702':  {'calibs': ['EC59048', 'EC59048', 'EC59050', 'EC59052', 'EC59054', 'EC59056', 'EC59058', 'EC59060', 'EC59062', 'EC59064', 'EC59067', 'EC59067', 'EC59069', 'EC59071', 'EC59073', 'EC59075', 'EC59078', 'EC59078', 'EC59080', 'EC59082', 'EC59084', 'EC59136', 'EC59136', 'EC59138', 'EC59141', 'EC59141', 'EC59143', 'EC59145', 'EC59147', 'EC59149', 'EC59151', 'EC59284'], 'REF_AP': '', 'objects': ['EC59047', 'EC59049', 'EC59051', 'EC59053', 'EC59055', 'EC59057', 'EC59059', 'EC59061', 'EC59063', 'EC59065', 'EC59066', 'EC59068', 'EC59070', 'EC59072', 'EC59074', 'EC59076', 'EC59077', 'EC59079', 'EC59081', 'EC59083', 'EC59085', 'EC59135', 'EC59137', 'EC59139', 'EC59140', 'EC59142', 'EC59144', 'EC59146', 'EC59148', 'EC59150', 'EC59152', 'EC59283'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201702/', 'ap_position': 'left', 'biases': ['EC59155', 'EC59156', 'EC59157', 'EC59158'], 'flats': ['EC59159', 'EC59160', 'EC59161', 'EC59162']} ,

  'binaries_13_201703':  {'calibs': ['EC59284', 'EC59288', 'EC59288', 'EC59292', 'EC59292', 'EC59294', 'EC59298', 'EC59298', 'EC59300', 'EC59303', 'EC59303', 'EC59305', 'EC59316', 'EC59316', 'EC59318', 'EC59335', 'EC59337', 'EC59337', 'EC59340', 'EC59340', 'EC59342', 'EC59382', 'EC59382', 'EC59384', 'EC59390', 'EC59390', 'EC59392', 'EC59401', 'EC59401'], 'REF_AP': '', 'objects': ['EC59285', 'EC59287', 'EC59289', 'EC59291', 'EC59293', 'EC59295', 'EC59297', 'EC59299', 'EC59301', 'EC59302', 'EC59304', 'EC59306', 'EC59315', 'EC59317', 'EC59319', 'EC59334', 'EC59336', 'EC59338', 'EC59339', 'EC59341', 'EC59343', 'EC59381', 'EC59383', 'EC59385', 'EC59389', 'EC59391', 'EC59393', 'EC59400', 'EC59402'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201703/', 'ap_position': 'left', 'biases': ['EC59326', 'EC59327', 'EC59328', 'EC59329', 'EC59330', 'EC59326', 'EC59327'], 'flats': ['EC59331', 'EC59332', 'EC59333'], 'JOIN':[['EC59315', 'EC59317', 'EC59319']]} ,

  'binaries_13_201704':  {'calibs': ['EC59429', 'EC59429', 'EC59431', 'EC59455', 'EC59457', 'EC59458', 'EC59460', 'EC59469', 'EC59469', 'EC59471', 'EC59474', 'EC59474', 'EC59476', 'EC59484', 'EC59484', 'EC59486', 'EC59489', 'EC59489', 'EC59491', 'EC59500', 'EC59500', 'EC59502', 'EC59505', 'EC59505'], 'REF_AP': '', 'objects': ['EC59428', 'EC59430', 'EC59432', 'EC59454', 'EC59456', 'EC59459', 'EC59461', 'EC59468', 'EC59470', 'EC59472', 'EC59473', 'EC59475', 'EC59477', 'EC59483', 'EC59485', 'EC59487', 'EC59488', 'EC59490', 'EC59492', 'EC59499', 'EC59501', 'EC59503', 'EC59504', 'EC59506'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201704/', 'ap_position': 'left', 'biases': ['EC59328', 'EC59329', 'EC59330', 'EC59521', 'EC59522', 'EC59523', 'EC59524', 'EC59525'], 'flats': ['EC59331', 'EC59332', 'EC59333', 'EC59518', 'EC59519', 'EC59520'], 'JOIN':[['EC59468', 'EC59470', 'EC59472']]} ,

  'binaries_13_201707':  {'calibs': ['EC59612', 'EC59612', 'EC59614', 'EC59623', 'EC59623', 'EC59625', 'EC59634', 'EC59634'], 'REF_AP': '', 'objects': ['EC59611', 'EC59613', 'EC59615', 'EC59622', 'EC59624', 'EC59626', 'EC59633', 'EC59635'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201707/', 'ap_position': 'left', 'biases': ['EC59627', 'EC59628', 'EC59629', 'EC59630', 'EC59631', 'EC59627', 'EC59628'], 'flats': ['EC59632', 'EC59636', 'EC59637']} ,

  #'binaries_13_201708':  {'calibs': [], 'REF_AP': '', 'objects': [], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201708/', 'ap_position': 'left', 'biases': ['EC59629', 'EC59630', 'EC59631'], 'flats': []} ,

  'binaries_13_201710':  {'calibs': ['EC59870', 'EC59870', 'EC59872', 'EC59875', 'EC59875', 'EC59877', 'EC59887', 'EC59905', 'EC59905', 'EC59907', 'EC59910', 'EC59910', 'EC59912', 'EC59915', 'EC59915', 'EC59917', 'EC59920', 'EC59944', 'EC59946', 'EC59946', 'EC59948', 'EC59951'], 'REF_AP': '', 'objects': ['EC59869', 'EC59871', 'EC59873', 'EC59874', 'EC59876', 'EC59878', 'EC59886', 'EC59904', 'EC59906', 'EC59908', 'EC59909', 'EC59911', 'EC59913', 'EC59914', 'EC59916', 'EC59918', 'EC59919', 'EC59943', 'EC59945', 'EC59947', 'EC59949', 'EC59950'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201710/', 'ap_position': 'center', 'biases': ['EC59921', 'EC59922', 'EC59923', 'EC59924', 'EC59925', 'EC59926', 'EC59927', 'EC59928', 'EC59929', 'EC60040', 'EC60041'], 'flats': ['EC59936', 'EC59937', 'EC59938']} ,

  'binaries_13_201712':  {'calibs': ['EC60052', 'EC60054', 'EC60054', 'EC60056', 'EC60059', 'EC60059', 'EC60061', 'EC60064', 'EC60064', 'EC60067', 'EC60067', 'EC60069', 'EC60072', 'EC60072', 'EC60074', 'EC60079', 'EC60085', 'EC60085', 'EC60087', 'EC60092', 'EC60104', 'EC60108', 'EC60108', 'EC60110', 'EC60113', 'EC60113', 'EC60115', 'EC60120', 'EC60120', 'EC60122', 'EC60127', 'EC60127', 'EC60129', 'EC60132', 'EC60132', 'EC60134'], 'REF_AP': '', 'objects': ['EC60051', 'EC60053', 'EC60055', 'EC60057', 'EC60058', 'EC60060', 'EC60062', 'EC60063', 'EC60065', 'EC60066', 'EC60068', 'EC60070', 'EC60071', 'EC60073', 'EC60075', 'EC60078', 'EC60084', 'EC60086', 'EC60088', 'EC60091', 'EC60103', 'EC60107', 'EC60109', 'EC60111', 'EC60112', 'EC60114', 'EC60116', 'EC60119', 'EC60121', 'EC60123', 'EC60126', 'EC60128', 'EC60130', 'EC60131', 'EC60133', 'EC60135'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201712/', 'ap_position': 'left', 'biases': ['EC60042', 'EC60043', 'EC60044'], 'flats': ['EC60046', 'EC60047', 'EC60048']} ,

  'binaries_13_201801':  {'calibs': ['EC60525', 'EC60525', 'EC60527', 'EC60531', 'EC60535', 'EC60535', 'EC60537', 'EC60542', 'EC60542', 'EC60670', 'EC60670', 'EC60674', 'EC60676', 'EC60676', 'EC60678', 'EC60681', 'EC60683', 'EC60685', 'EC60685', 'EC60687', 'EC60690', 'EC60692', 'EC60693', 'EC60695', 'EC60698', 'EC60698', 'EC60700', 'EC60703', 'EC60703', 'EC60705', 'EC60708', 'EC60708', 'EC60710', 'EC60713', 'EC60715', 'EC60717'], 'REF_AP': '', 'objects': ['EC60524', 'EC60526', 'EC60528', 'EC60530', 'EC60534', 'EC60536', 'EC60538', 'EC60541', 'EC60543', 'EC60669', 'EC60671', 'EC60673', 'EC60675', 'EC60677', 'EC60679', 'EC60680', 'EC60682', 'EC60684', 'EC60686', 'EC60688', 'EC60689', 'EC60691', 'EC60694', 'EC60696', 'EC60697', 'EC60699', 'EC60701', 'EC60702', 'EC60704', 'EC60706', 'EC60707', 'EC60709', 'EC60711', 'EC60712', 'EC60714', 'EC60716'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201801/', 'ap_position': 'center', 'biases': ['EC60718', 'EC60719', 'EC60720', 'EC60721', 'EC60722', 'EC60723', 'EC60724'], 'flats': ['EC60046', 'EC60047', 'EC60048'], 'JOIN':[['EC60684', 'EC60686', 'EC60688']]} ,

  #'binaries_13_201803':  {'calibs': [], 'REF_AP': '', 'objects': [], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201803/', 'ap_position': 'left', 'biases': ['EC60725', 'EC60726', 'EC60727'], 'flats': ['EC60728', 'EC60729', 'EC60730']} ,

  'binaries_13_201804':  {'calibs': ['EC60734', 'EC60734', 'EC60797', 'EC60797', 'EC60799', 'EC60802', 'EC60802', 'EC60804', 'EC60807', 'EC60807', 'EC60811', 'EC60811', 'EC60813', 'EC60824', 'EC60824', 'EC60826', 'EC60829', 'EC60829', 'EC60831', 'EC60844', 'EC60882', 'EC60882', 'EC60886', 'EC60886', 'EC60888', 'EC60891', 'EC60891', 'EC60909', 'EC60909', 'EC60911', 'EC60920'], 'REF_AP': '', 'objects': ['EC60733', 'EC60735', 'EC60796', 'EC60798', 'EC60800', 'EC60801', 'EC60803', 'EC60805', 'EC60806', 'EC60808', 'EC60810', 'EC60812', 'EC60814', 'EC60823', 'EC60825', 'EC60827', 'EC60828', 'EC60830', 'EC60832', 'EC60843', 'EC60881', 'EC60883', 'EC60885', 'EC60887', 'EC60889', 'EC60890', 'EC60892', 'EC60908', 'EC60910', 'EC60912', 'EC60919'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201804/', 'ap_position': 'center', 'biases': ['EC60791', 'EC60792', 'EC60793', 'EC60794', 'EC60795', 'EC60859', 'EC60860', 'EC60861', 'EC60862', 'EC60863', 'EC60944', 'EC60945'], 'flats': ['EC60752', 'EC60753', 'EC60754', 'EC60820', 'EC60821', 'EC60822', 'EC60864', 'EC60865', 'EC60866', 'EC60867', 'EC60868'], 'JOIN':[['EC60881', 'EC60883']]} ,

  'binaries_13_201806':  {'calibs': ['EC60979', 'EC60987', 'EC60987', 'EC61010', 'EC61010'], 'REF_AP': '', 'objects': ['EC60978', 'EC60986', 'EC60988', 'EC61009', 'EC61011'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201806/', 'ap_position': 'center', 'biases': ['EC60946', 'EC60947', 'EC60948'], 'flats': ['EC60949', 'EC60950', 'EC60951', 'EC60952', 'EC60953']} ,

  #'binaries_13_201807':  {'calibs': [], 'REF_AP': '', 'objects': [], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201807/', 'ap_position': 'left', 'biases': ['EC61155', 'EC61156', 'EC61157', 'EC61158', 'EC61159'], 'flats': ['EC61160', 'EC61161', 'EC61162']} ,

  'binaries_13_201808':  {'calibs': ['EC61213', 'EC61213', 'EC61220', 'EC61220', 'EC61222', 'EC61249', 'EC61249', 'EC61251', 'EC61283', 'EC61283', 'EC61285'], 'REF_AP': '', 'objects': ['EC61212', 'EC61214', 'EC61219', 'EC61221', 'EC61223', 'EC61248', 'EC61250', 'EC61252', 'EC61282', 'EC61284', 'EC61286'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201808/', 'ap_position': 'center', 'biases': ['EC61277', 'EC61278', 'EC61279', 'EC61280', 'EC61281'], 'flats': ['EC61289', 'EC61290', 'EC61291', 'EC61292', 'EC61293'], 'JOIN':[['EC61248', 'EC61250', 'EC61252']]} ,

  'binaries_13_201809':  {'calibs': ['EC61336', 'EC61336', 'EC61338', 'EC61346', 'EC61346', 'EC61351', 'EC61351', 'EC61353', 'EC61385', 'EC61385', 'EC62031'], 'REF_AP': '', 'objects': ['EC61335', 'EC61337', 'EC61339', 'EC61345', 'EC61347', 'EC61350', 'EC61352', 'EC61354', 'EC61384', 'EC61386', 'EC62030'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201809/', 'ap_position': 'center', 'biases': ['EC61355', 'EC61356', 'EC61357', 'EC61358', 'EC61359', 'EC61360', 'EC61361'], 'flats': ['EC61362', 'EC61363', 'EC61364', 'EC61365']} ,

  'binaries_13_201812':  {'calibs': ['EC62031', 'EC62033', 'EC62036', 'EC62036', 'EC62038', 'EC62057', 'EC62057', 'EC62061', 'EC62061', 'EC62063', 'EC62066', 'EC62066', 'EC62074', 'EC62074', 'EC62078', 'EC62078', 'EC62082', 'EC62082', 'EC62084', 'EC62103', 'EC62103', 'EC62105', 'EC62289'], 'REF_AP': '', 'objects': ['EC62032', 'EC62034', 'EC62035', 'EC62037', 'EC62039', 'EC62056', 'EC62058', 'EC62060', 'EC62062', 'EC62064', 'EC62065', 'EC62067', 'EC62073', 'EC62075', 'EC62077', 'EC62079', 'EC62081', 'EC62083', 'EC62085', 'EC62102', 'EC62104', 'EC62106', 'EC62288'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201812/', 'ap_position': 'center', 'biases': ['EC62044', 'EC62045', 'EC62046', 'EC62047', 'EC62048', 'EC62049', 'EC62050'], 'flats': ['EC62051', 'EC62052', 'EC62053', 'EC62054', 'EC62055']} ,

  'binaries_13_201901':  {'calibs': ['EC62289', 'EC62291', 'EC62297', 'EC62297', 'EC62299', 'EC62314', 'EC62314', 'EC62316', 'EC62319', 'EC62319', 'EC62321', 'EC62337', 'EC62337', 'EC62339', 'EC62350', 'EC62350', 'EC62352', 'EC62355', 'EC62355', 'EC62357', 'EC62366', 'EC62366', 'EC62372', 'EC62372', 'EC62374', 'EC62377', 'EC62377', 'EC62379', 'EC62382', 'EC62382', 'EC62384', 'EC62387', 'EC62387', 'EC62389', 'EC62392', 'EC62392', 'EC62394', 'EC62441'], 'REF_AP': '', 'objects': ['EC62290', 'EC62292', 'EC62296', 'EC62298', 'EC62300', 'EC62313', 'EC62315', 'EC62317', 'EC62318', 'EC62320', 'EC62322', 'EC62336', 'EC62338', 'EC62340', 'EC62349', 'EC62351', 'EC62353', 'EC62354', 'EC62356', 'EC62358', 'EC62365', 'EC62367', 'EC62371', 'EC62373', 'EC62375', 'EC62376', 'EC62378', 'EC62380', 'EC62381', 'EC62383', 'EC62385', 'EC62386', 'EC62388', 'EC62390', 'EC62391', 'EC62393', 'EC62395', 'EC62440'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201901/', 'ap_position': 'center', 'biases': ['EC62399', 'EC62400', 'EC62401', 'EC62402', 'EC62403'], 'flats': ['EC62396', 'EC62397', 'EC62398'], 'JOIN':[['EC62318', 'EC62320', 'EC62322']]} ,

  'binaries_13_201902':  {'calibs': ['EC62441', 'EC62443', 'EC62452', 'EC62452', 'EC62473', 'EC62473', 'EC62475', 'EC62478', 'EC62478', 'EC62480', 'EC62483', 'EC62483', 'EC62486', 'EC62486', 'EC62488', 'EC62491', 'EC62491', 'EC62494', 'EC62494', 'EC62503', 'EC62503', 'EC62505', 'EC62519', 'EC62519', 'EC62521', 'EC62524', 'EC62524', 'EC62527', 'EC62527', 'EC62529', 'EC62532', 'EC62532', 'EC62534', 'EC62537', 'EC62537', 'EC62539', 'EC62546', 'EC62546', 'EC62548'], 'REF_AP': '', 'objects': ['EC62442', 'EC62444', 'EC62451', 'EC62453', 'EC62472', 'EC62474', 'EC62476', 'EC62477', 'EC62479', 'EC62481', 'EC62482', 'EC62484', 'EC62485', 'EC62487', 'EC62489', 'EC62490', 'EC62492', 'EC62493', 'EC62495', 'EC62502', 'EC62504', 'EC62506', 'EC62518', 'EC62520', 'EC62522', 'EC62523', 'EC62525', 'EC62526', 'EC62528', 'EC62530', 'EC62531', 'EC62533', 'EC62535', 'EC62536', 'EC62538', 'EC62540', 'EC62545', 'EC62547', 'EC62549'], 'REF_ARC': '../../ref_wav_files/', 'ORIG_DIR': '../201902/', 'ap_position': 'center', 'biases': ['EC62462', 'EC62463', 'EC62464', 'EC62465', 'EC62466', 'EC62467', 'EC62468'], 'flats': ['EC62469', 'EC62470', 'EC62471']} ,

  'binaries_13_201907':  {'objects': ['EC63196', 'EC63198', 'EC63200', 'EC63206', 'EC63208', 'EC63210', 'EC63211', 'EC63213', 'EC63217', 'EC63219', 'EC63228', 'EC63230', 'EC63232', 'EC63233', 'EC63235', 'EC63251', 'EC63253', 'EC63257', 'EC63259', 'EC63275', 'EC63277'], 
                          'calibs': ['EC63197', 'EC63197', 'EC63199', 'EC63207', 'EC63209', 'EC63209', 'EC63212', 'EC63212', 'EC63218', 'EC63220', 'EC63229', 'EC63231', 'EC63231', 'EC63234', 'EC63236', 'EC63252', 'EC63254', 'EC63258', 'EC63260', 'EC63276', 'EC63278'], 
                          'biases': ['EC63188', 'EC63189', 'EC63190', 'EC63191', 'EC63192'], 
                          'flats': ['EC63193', 'EC63194', 'EC63195'],
                          'REF_AP': '',                           
                          'REF_ARC': '../../ref_wav_files/', 
                          'ORIG_DIR': '../201907/', 
                          'ap_position': 'center'},

  'binaries_13_201908':  {'objects': ['EC63283', 'EC63285', 'EC63287'], 
                          'calibs': ['EC63284', 'EC63286', 'EC63286'], 
                          'REF_AP': '',                           
                          'REF_ARC': '../../ref_wav_files/', 
                          'ORIG_DIR': '../201908/', 
                          'ap_position': 'center', 
                          'biases': ['EC63290', 'EC63291', 'EC63292', 'EC63293', 'EC63294'], 
                          'flats': ['EC63193', 'EC63194', 'EC63195']} , # flats from previous run

  'binaries_13_201912':  {'objects': ['EC63463', 'EC63465', 'EC63471', 'EC63473', 'EC63475', 'EC63476', 'EC63501', 'EC63503', 'EC63513', 'EC63515', 'EC63526', 'EC63530', 'EC63532', 'EC63552', 'EC63554', 'EC63556', 'EC63558', 'EC63560', 'EC63574', 'EC63576', 'EC63580', 'EC63582', 'EC63586', 'EC63588', 'EC63590', 'EC63592', 'EC63598', 'EC63600', 'EC63608', 'EC63610', 'EC63612', 'EC63614', 'EC63616', 'EC63618', 'EC63620', 'EC63634', 'EC63636', 'EC63638', 'EC63640', 'EC63642', 'EC63644', 'EC63646', ], 
                          'calibs': ['EC63464', 'EC63466', 'EC63472', 'EC63474', 'EC63474', 'EC63477', 'EC63502', 'EC63502', 'EC63514', 'EC63516', 'EC63527', 'EC63531', 'EC63531', 'EC63553', 'EC63555', 'EC63557', 'EC63559', 'EC63561', 'EC63575', 'EC63577', 'EC63581', 'EC63583', 'EC63587', 'EC63589', 'EC63591', 'EC63593', 'EC63599', 'EC63599', 'EC63609', 'EC63611', 'EC63613', 'EC63615', 'EC63617', 'EC63619', 'EC63621', 'EC63635', 'EC63637', 'EC63639', 'EC63641', 'EC63643', 'EC63645', 'EC63647'], 
                          'REF_AP': '',                           
                          'REF_ARC': '../../ref_wav_files/', 
                          'ORIG_DIR': '../201912/', 
                          'ap_position': 'center', 
                          'biases': ['EC63504', 'EC63505', 'EC63506', 'EC63507', 'EC63508', 'EC63509', 'EC63564', 'EC63565', 'EC63566', 'EC63567', 'EC63568', 'EC63653', 'EC63654', 'EC63655', 'EC63656', 'EC63657'], # ekstra biases? 
                          'flats': ['EC63510', 'EC63511', 'EC63512', 'EC63569', 'EC63570', 'EC63571', 'EC63572', 'EC63573', 'EC63596', 'EC63597', 'EC63648', 'EC63649', 'EC63650', 'EC63651', 'EC63652']} , # ekstra flati?

  'binaries_13_202001':  {'objects': ['EC63907', 'EC63909', 'EC63911', 'EC63913', 'EC63915', 'EC63917', 'EC63919', 'EC63921', 'EC63979', 'EC63981', 'EC63983', 'EC63991', 'EC63993', 'EC63995', 'EC64005', 'EC64051', 'EC64053', 'EC64055', 'EC64057'], 
                          'REF_AP': '', 
                          'calibs': ['EC63908', 'EC63910', 'EC63912', 'EC63914', 'EC63916', 'EC63918', 'EC63920', 'EC63922', 'EC63980', 'EC63982', 'EC63984', 'EC63992', 'EC63994', 'EC63996', 'EC64006', 'EC64052', 'EC64054', 'EC64056', 'EC64058'], 
                          'REF_ARC': '../../ref_wav_files/', 
                          'ORIG_DIR': '../202001/', 
                          'ap_position': 'center', 
                          'biases': ['EC63928', 'EC63929', 'EC63930', 'EC63931'], # is this correct, look at log ??? 
                          'flats': ['EC63923', 'EC63924', 'EC63925', 'EC63926']} , # wtf logs, why so many???

  'binaries_13_202002':  {'objects': ['EC64540', 'EC64542', 'EC64544', 'EC64560', 'EC64562', 'EC64630', 'EC64632', 'EC64634', 'EC64636', 'EC64638', 'EC64640', 'EC64642', 'EC64681', 'EC64683', 'EC64685', 'EC64687', 'EC64689', 'EC64691', 'EC64693'], 
                          'REF_AP': '', 
                          'calibs': ['EC64541', 'EC64543', 'EC64545', 'EC64561', 'EC64563', 'EC64631', 'EC64633', 'EC64635', 'EC64637', 'EC64639', 'EC64641', 'EC64643', 'EC64682', 'EC64684', 'EC64686', 'EC64688', 'EC64690', 'EC64692', 'EC64694'], 
                          'REF_ARC': '../../ref_wav_files/', 
                          'ORIG_DIR': '../202002/', 
                          'ap_position': 'center', 
                          'biases': ['EC64595', 'EC64596', 'EC64597', 'EC64598', 'EC64599'], # bias/flat every night???
                          'flats': ['EC64590', 'EC64591', 'EC64592', 'EC64593', 'EC64594']} ,

}

for obs_key in observations.keys():
	objects = observations[obs_key]['objects']

	source_dir = reduc_dir + obs_key + '/'

	for ec in objects:
		source_fits1 = source_dir + ec +'.ec.fits'
		source_fits2 = source_dir + ec +'.ec.vh.fits'
		date_f = obs_key.split('_')[-1]

		if not os.path.isfile(source_fits1):
			print(ec + ' - no data (' + date_f + ')')
			continue

		if not os.path.isfile(source_fits2):
			print(ec + ' - no vh   (' + date_f + ')')
			continue

		os.system('rm temp.txt')
		iraf.wspectext(input=source_fits2+'[*,15,1]', output='temp.txt', header='no')
		s = np.loadtxt('temp.txt')
		wvl = s[:, 0]
		if wvl[1] - wvl[0] == 1:
			print(ec + ' - no wvl  (' + date_f + ')')


raise SystemExit

obs_metadata = Table.read(data_dir + 'star_data_all.csv')
obs_metadata = obs_metadata[obs_metadata['odkdaj'] == 'nova']

_go_to_dir('Binaries_spectra')
copyfile(data_dir + 'star_data_all.csv', 'star_data_all.csv')

for star in ['TV_LMi', 'GZ_Dra', 'V455_Aur', 'GK_Dra', 'V1898_Cyg', 'V417_Aur', 'DT_Cam', 'V394_Vul', 'CI_CVn', 'EQ_Boo', 'V994_Her', 'CN_Lyn', 'DV_Cam']:

	print('Working on star ' + star)
	star_obs = obs_metadata[obs_metadata['star'] == star.replace('_', ' ').lower()]

	source_folder = ['binaries_13_' + dd[:4] + dd[5:7] for dd in star_obs['dateobs']]
	star_obs['source_folder'] = source_folder

	print(star_obs[np.argsort(star_obs['JD'])])

	_go_to_dir(star)

	# remove everything
	os.system('rm -R *')

	_go_to_dir('spec')

	n_no_spec = 0
	n_no_wvl = 0
	for i_s in range(len(star_obs)):
		star_spec = star_obs[i_s]
		spec_name = star_spec['filename']
		spec_suff = '.ec.vh'
		print(spec_name)

		# does reduced data exist
		targ_dir = reduc_dir + star_spec['source_folder'] + '/' + spec_name + spec_suff + '.fits'
		if not os.path.isfile(targ_dir):
			print(' Not found ' + targ_dir)
			n_no_spec += 1
			continue

		# copy reduced data
		copyfile(targ_dir, spec_name + spec_suff + '.fits')
		_go_to_dir(spec_name + spec_suff)
		export_observation_to_txt(targ_dir, spec_name + spec_suff)

		# normalise spectra
		print(' Normalising spectra')
		bad_wvl_order = 0
		no_data_order = 0
		for txt_file in glob(spec_name + '*_*.txt'):
			txt_out = txt_file[:-4] + '_normalised.txt'
			order_data = np.loadtxt(txt_file)

			if len(order_data) == 0:
				print(' No data in order ' + txt_file)
				no_data_order += 1
				continue

			# crop order data to remove noisy part of the echelle order
			order_data = order_data[100:-100, :]
			n_data = order_data.shape[0]

			ref_flx_norm_curve1 = _spectra_normalize(np.arange(n_data), order_data[:, 1],
													steps=10, sigma_low=2., sigma_high=2.5, n_min_perc=8.,
													order=11, func='cheb', return_fit=True, median_init=False)

			ref_flx_norm_curve2 = _spectra_normalize(np.arange(n_data), order_data[:, 1]/ref_flx_norm_curve1,
													steps=10, sigma_low=2., sigma_high=2.5, n_min_perc=8.,
													order=3, func='cheb', return_fit=True, median_init=True)

			# renorm order
			fig, ax = plt.subplots(3, 1, sharex=True, figsize=(13, 7))
			ax[0].plot(order_data[:, 0], order_data[:, 1])
			ax[0].plot(order_data[:, 0], ref_flx_norm_curve1)
			ax[1].plot(order_data[:, 0], order_data[:, 1] / ref_flx_norm_curve1)
			ax[1].plot(order_data[:, 0], ref_flx_norm_curve2)
			ax[2].plot(order_data[:, 0], order_data[:, 1] / ref_flx_norm_curve1 / ref_flx_norm_curve2)
			ax[1].set(ylim=(0.3, 1.2))
			ax[2].set(ylim=(0.3, 1.2), xlim=(order_data[0, 0], order_data[-1, 0]))
			fig.tight_layout()
			fig.subplots_adjust(hspace=0, wspace=0)
			fig.savefig(txt_file[:-4] + '_normalised.png')
			plt.close(fig)

			if order_data[1, 0] - order_data[0, 0] == 1:
				print(' No wav cal in ' + txt_file)
				bad_wvl_order += 1
				continue

			order_data[:, 1] = order_data[:, 1] / ref_flx_norm_curve1
			np.savetxt(txt_out, order_data, fmt=['%.5f', '%.5f'])

		if bad_wvl_order > 20:
			n_no_wvl += 1

		os.chdir('..')

	print("Stats for {:s} are  ->  exposures: {:.0f}  no-data: {:.0f}  no-wvl: {:.0f}  ok: {:.0f}".format(star, len(star_obs), n_no_spec, n_no_wvl, len(star_obs) - n_no_wvl - n_no_spec))

	os.chdir('..')
	os.chdir('..')
	print('==========================================================================')
	print('==========================================================================')
	print('\n \n \n')
