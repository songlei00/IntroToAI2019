/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package controllers.learningmodel;

import core.game.Observation;
import core.game.StateObservation;
import tools.Vector2d;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.LinkedList;

/**
 *
 * @author yuy
 */
public class RLDataExtractor {
    public FileWriter filewriter;
    public static Instances s_datasetHeader = datasetHeader();
    
    public RLDataExtractor(String filename) throws Exception{

        filewriter = new FileWriter(filename+".arff");
        filewriter.write(s_datasetHeader.toString());

        /*
                // ARFF File header
        filewriter.write("@RELATION AliensData\n");
        // Each row denotes the feature attribute
        // In this demo, the features have four dimensions.
        filewriter.write("@ATTRIBUTE gameScore  NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarSpeed  NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarHealthPoints NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarType NUMERIC\n");
        // objects
        for(int y=0; y<14; y++)
            for(int x=0; x<32; x++)
                filewriter.write("@ATTRIBUTE object_at_position_x=" + x + "_y=" + y + " NUMERIC\n");
        // The last row of the ARFF header stands for the classes
        filewriter.write("@ATTRIBUTE Class {0,1,2}\n");
        // The data will recorded in the following.
        filewriter.write("@Data\n");*/
        
    }
    
    public static Instance makeInstance(double[] features, int action, double reward){
        features[430] = action;
        features[431] = reward;
        Instance ins = new Instance(1, features);
        ins.setDataset(s_datasetHeader);
        return ins;
    }
    
    public static double[] featureExtract(StateObservation obs){

        double[] feature = new double[432];  // 420 + 4 + 1(action) + 1(Q) 432
        boolean isTopBlock = false;
        boolean isDangerous = false;
        double Manhattan = 0;

        // 420 locations
        int[][] map = new int[28][15];
        // Extract features
        LinkedList<Observation> allobj = new LinkedList<>();
        if( obs.getImmovablePositions()!=null )
            for(ArrayList<Observation> l : obs.getImmovablePositions()) allobj.addAll(l);
        if( obs.getMovablePositions()!=null )
            for(ArrayList<Observation> l : obs.getMovablePositions()) allobj.addAll(l);
        if( obs.getPortalsPositions()!=null)
            for(ArrayList<Observation> l : obs.getPortalsPositions()) allobj.addAll(l); // only one object

        Vector2d pos = obs.getAvatarPosition();
        pos.x /= 28;
        pos.y /= 28;

        for(Observation o : allobj){
            Vector2d p = o.position;
            int x = (int)(p.x/28);
            int y= (int)(p.y/28);
            map[x][y] = o.itype;
            if(o.itype == 13 && x == pos.x && y-pos.y>=-3 && y-pos.y <=-1)
                isTopBlock = true;
            else if( (o.itype == 7 || o.itype == 8 || o.itype == 10 || o.itype == 11) &&
                    Math.abs(x - pos.x) < 4 && y-pos.y>=-2 && y-pos.y <=0 )
                isDangerous = true;
            else if(o.itype == 4)
                Manhattan = Math.abs(x-pos.x) + Math.abs(y-pos.y);
        }

        for(int y=0; y<15; y++)
            for(int x=0; x<28; x++)
                feature[y*28+x] = map[x][y];

        feature[420] = pos.x;
        feature[421] = pos.y;
        feature[422] = isTopBlock ? 1 : 0;
        feature[423] = isDangerous ? 1 : 0;
        feature[424] = -1*Manhattan; // Max???
        feature[425] = obs.getGameScore();
        feature[426] = obs.getGameTick();
        feature[427] = obs.getAvatarHealthPoints();
        feature[428] = obs.getAvatarType();
        //System.out.printf("pos_x: %f, pos_y: %f\n", pos.x, pos.y);
        //System.out.printf("%d, %d\n", isTopBlock ? 1 : 0, isDangerous ? 1 : 0);
        //System.out.printf("%f\n", Manhattan);


        return feature;
    }
    
    public static Instances datasetHeader(){
        
        if (s_datasetHeader!=null)
            return s_datasetHeader;
        
        FastVector attInfo = new FastVector();
        // 420 locations
        for(int y=0; y<15; y++){
            for(int x=0; x<28; x++){
                Attribute att = new Attribute("object_at_position_x=" + x + "_y=" + y);
                attInfo.addElement(att);
            }
        }
        Attribute att = new Attribute("Pos_x" ); attInfo.addElement(att);
        att = new Attribute("Pos_y" ); attInfo.addElement(att);
        att = new Attribute("isTopBlock" ); attInfo.addElement(att);
        att = new Attribute("isDangerous" ); attInfo.addElement(att);
        att = new Attribute("Manhattan" ); attInfo.addElement(att);
        att = new Attribute("GameScore" ); attInfo.addElement(att);
        att = new Attribute("GameTick" ); attInfo.addElement(att);
        att = new Attribute("AvatarHealthPoints" ); attInfo.addElement(att);
        att = new Attribute("AvatarType" ); attInfo.addElement(att);
        //action
        FastVector actions = new FastVector();
        actions.addElement("0");
        actions.addElement("1");
        actions.addElement("2");
        actions.addElement("3");
        att = new Attribute("actions", actions);        
        attInfo.addElement(att);
        // Q value
        att = new Attribute("Qvalue");
        attInfo.addElement(att);
        
        Instances instances = new Instances("PacmanQdata", attInfo, 0);
        instances.setClassIndex( instances.numAttributes() - 1);
        
        return instances;
    }
    
}
