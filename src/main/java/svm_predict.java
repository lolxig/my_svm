import libsvm.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

class svm_predict {
    private static final svm_print_interface svm_print_null = s -> {
    };

    private static final svm_print_interface svm_print_stdout = System.out::print;

    private static svm_print_interface svm_print_string = svm_print_stdout;

    /**
     * 打印信息.
     */
    static void info(String s) {
        svm_print_string.print(s);
    }

    /**
     * 字符串转换为double.
     */
    private static double atof(String s) {
        return Double.parseDouble(s);
    }

    /**
     * 字符串转换为int.
     */
    private static int atoi(String s) {
        return Integer.parseInt(s);
    }

    /**
     * 进行数据预测.
     *
     * @param input               输入数据流
     * @param output              输出数据流
     * @param model               训练好的模型
     * @param predict_probability 是否进行概率预测
     */
    private static void predict(BufferedReader input, DataOutputStream output,
                                svm_model model, int predict_probability) throws IOException {
        int correct = 0;    //正确分类的个数
        int total = 0;      //总个数
        double error = 0;   //错分类个数
        double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

        int svm_type = svm.svm_get_svm_type(model); //获取SVM的类型
        int nr_class = svm.svm_get_nr_class(model); //获取类别个数
        double[] prob_estimates = null;

        //如果需要进行概率预测
        if (predict_probability == 1) {
//            //回归
//            if (svm_type == svm_parameter.EPSILON_SVR ||
//                    svm_type == svm_parameter.NU_SVR) {
//                svm_predict.info("Prob. model for test data: target value = predicted value + z,\n" +
//                        "z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma="
//                        + svm.svm_get_svr_probability(model) + "\n");
//                //分类
//            } else {
//                int[] labels = new int[nr_class];
//                svm.svm_get_labels(model, labels);
//                prob_estimates = new double[nr_class];
//                output.writeBytes("labels");
//                for (int j = 0; j < nr_class; j++)
//                    output.writeBytes(" " + labels[j]);
//                output.writeBytes("\n");
//            }
        }
        //逐行读取，开始预测
        while (true) {
            String line = input.readLine();
            if (line == null) break;

            String[] fields = line.split("\t", -1);
            //填充标签
            double target_label = atof(fields[fields.length - 1]);
            //填充特征向量
            svm_node[] x = new svm_node[fields.length - 1];
            for (int i = 0; i < fields.length - 1; i++) {
                x[i] = new svm_node();
                x[i].index = i + 1;
                x[i].value = atof(fields[i]);
            }

//            StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
//
//            //读取目标标签
//            double target_label = atof(st.nextToken());
//            //读取特征集
//            int m = st.countTokens() / 2;
//            svm_node[] x = new svm_node[m];
//            for (int j = 0; j < m; j++) {
//                x[j] = new svm_node();
//                x[j].index = atoi(st.nextToken());
//                x[j].value = atof(st.nextToken());
//            }

            //预测值
            double predict_label = 0;
            if (predict_probability == 1 && (svm_type == svm_parameter.C_SVC || svm_type == svm_parameter.NU_SVC)) {
                //进行概率预测
//                predict_label = svm.svm_predict_probability(model, x, prob_estimates);
//                output.writeBytes(predict_label + " ");
//                for (int j = 0; j < nr_class; j++)
//                    output.writeBytes(prob_estimates[j] + " ");
//                output.writeBytes("\n");
            } else {
                //进行非概率预测
                predict_label = svm.svm_predict(model, x);
                output.writeBytes(predict_label + "\n");
            }

            //正确分类
            if (predict_label == target_label)
                ++correct;
            //分类误差平方
            error += (predict_label - target_label) * (predict_label - target_label);
            sump += predict_label;  //预测总值
            sumt += target_label;   //目标总值
            sumpp += predict_label * predict_label; //预测值平方和
            sumtt += target_label * target_label;   //目标值平方和
            sumpt += predict_label * target_label;  //目标值和预测值的平方和
            ++total;
        }
            //回归模型
        if (svm_type == svm_parameter.EPSILON_SVR ||
                svm_type == svm_parameter.NU_SVR) {
            svm_predict.info("Mean squared error = " + error / total + " (regression)\n");
            svm_predict.info("Squared correlation coefficient = " +
                    ((total * sumpt - sump * sumt) * (total * sumpt - sump * sumt)) /
                            ((total * sumpp - sump * sump) * (total * sumtt - sumt * sumt)) +
                    " (regression)\n");
            //分类模型
        } else
            svm_predict.info("Accuracy = " + (double) correct / total * 100 +
                    "% (" + correct + "/" + total + ") (classification)\n");
    }

    /**
     * 打印帮助信息.
     */
    private static void exit_with_help() {
        System.err.print(
                //输入参数，分别是    [选项] 待测文件 模型文件 输出文件
                "usage: svm_predict [options] test_file model_file output_file\n"
                        + "options:\n"
                        //概率估计，默认不进行概率估计
                        + "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n"
                        //退出模式，没有打印输出
                        + "-q : quiet mode (no outputs)\n");
        System.exit(1);
    }

    public void run(String[] argv) throws IOException {
        int i, predict_probability = 0;
        //输入命令行参数解析
        for (i = 0; i < argv.length; i++) {
            if (argv[i].charAt(0) != '-')
                break;
            ++i;
            switch (argv[i - 1].charAt(1)) {
                case 'b':
                    predict_probability = atoi(argv[i]);
                    break;
                case 'q':
                    svm_print_string = svm_print_null;
                    i--;
                    break;
                default:
                    System.err.print("Unknown option: " + argv[i - 1] + "\n");
                    exit_with_help();
            }
        }
        //如果参数不对，打印帮助信息并退出
        if (i >= argv.length - 2)
            exit_with_help();

        try {
            String inputFilePath = argv[i];
            String modelFilePath = argv[i + 1];
            String outputFilePath = argv[i + 2];

            BufferedReader input = new BufferedReader(new FileReader(inputFilePath));
            DataOutputStream output = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(outputFilePath)));
            //加载模型
            svm_model model = svm.svm_load_model(modelFilePath);
            //必须有模型文件
            if (model == null) {
                System.err.print("can't open model file " + modelFilePath + "\n");
                System.exit(1);
            }
            //检查是否预测概率
            if (predict_probability == 1) {
                if (!svm.svm_check_probability_model(model)) {
                    System.err.print("Model does not support probability estimates\n");
                    System.exit(1);
                }
            } else {
                if (svm.svm_check_probability_model(model)) {
                    svm_predict.info("Model supports probability estimates, but disabled in prediction.\n");
                }
            }
            //进行预测
            predict(input, output, model, predict_probability);
            input.close();
            output.close();
        } catch (FileNotFoundException | ArrayIndexOutOfBoundsException e) {
            exit_with_help();
        }
    }

    /**
     * 入口类.
     */
    public static void main(String[] argv) throws IOException {

        List<String[]> paramList = new ArrayList<>();
        try (BufferedReader in = new BufferedReader(
                new InputStreamReader(new FileInputStream("local_param_predict"), StandardCharsets.UTF_8))) {
            String line;
            while ((line = in.readLine()) != null) {
                line = line.trim().replaceAll(" +", " ");
                paramList.add(line.split(" "));
            }
        }


//        FileReader in = new FileReader("local_param_predict");
//        char[] buff = new char[1024];
//        int len = in.read(buff);
//        //输入命令行参数
//        String inputParams = new String(buff, 0, len);
//        //切割命令行参数
//        argv = inputParams.split(" ");

        //默认打印函数
        svm_print_string = svm_print_stdout;

        for (String[] params : paramList) {
            svm_predict predict = new svm_predict();
            predict.run(params);
        }

    }
}
