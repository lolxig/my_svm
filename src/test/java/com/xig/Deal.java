package com.xig;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/**
 * Created with IntelliJ IDEA.
 * Date: 2020-09-02 22:29
 * Author: nullpo
 * Description:
 */
public class Deal {

    public static final String INPATH = "E:\\论文\\小论文两篇\\IDS实验\\sample.csv";
    public static final String OUTPATH = "E:\\论文\\小论文两篇\\IDS实验\\out.csv";

    private static final int BUFF_SZ = 512 * 1024;

    public static void main(String[] args) throws Exception {

        try (
                BufferedReader in = new BufferedReader(
                        new InputStreamReader(new FileInputStream(INPATH), StandardCharsets.UTF_8), BUFF_SZ);
                BufferedWriter out = new BufferedWriter(
                        new OutputStreamWriter(new FileOutputStream(OUTPATH), StandardCharsets.UTF_8), BUFF_SZ)
        ) {
            out.write("");
            Map<String, String> map = new HashMap<String, String>() {{
                put("Benign", "1");
                put("DDOS attack-HOIC", "2");
                put("Bot", "3");
                put("Infilteration", "4");
                put("SSH-Bruteforce", "5");
                put("FTP-BruteForce", "6");
                put("DoS attacks-GoldenEye", "7");
                put("DoS attacks-Hulk", "8");
                put("DoS attacks-SlowHTTPTest", "9");
                put("DoS attacks-Slowloris", "10");
                put("Brute Force -Web", "11");
                put("Brute Force -XSS", "12");
                put("SQL Injection", "13");
            }};

            String line;
            while ((line = in.readLine()) != null) {
                String[] elems = line.split("\t", -1);
                if (!check(elems)) {
                    continue;
                }
//                out.append(line).append("\n");
                String label = elems[elems.length - 1];
                if (!map.containsKey(label)) {
                    System.out.println(line);
                    continue;
                }
                elems[elems.length - 1] = map.get(label);
                out.append(String.join("\t", elems)).append("\n");
            }

        }


    }

    private static boolean check(String[] elems) {
        for (String elem : elems) {
            if (elem.equals("NaN") || elem.equals("Infinity"))
                return false;
        }
        return true;
    }

}
