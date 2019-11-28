/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tools;

import core.game.Observation;
import core.game.StateObservation;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Observable;

import ontology.Types;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;

/**
 *
 * @author yuy
 */
public class Recorder {
    public FileWriter filewriter;
    public static Instances s_datasetHeader = datasetHeader();
    
    public Recorder(String filename) throws Exception{
        
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
    
    public static double[] featureExtract(StateObservation obs){

        int ava_x = 0;
        int ava_y = 0;
        boolean dangerous = false;

        double[] feature = new double[454];  // 448 + Tick + Score + x + y + dangerous + 1(class)
        
        // 448 locations
        int[][] map = new int[32][14];
        // Extract features
        LinkedList<Observation> allobj = new LinkedList<>();
        if( obs.getImmovablePositions()!=null )
            for(ArrayList<Observation> l : obs.getImmovablePositions()) allobj.addAll(l);
        if( obs.getMovablePositions()!=null )
            for(ArrayList<Observation> l : obs.getMovablePositions()) allobj.addAll(l);
        if( obs.getNPCPositions()!=null )
            for(ArrayList<Observation> l : obs.getNPCPositions()) allobj.addAll(l);

        if(obs.getAvatarPosition()!=null){
            Vector2d pos = obs.getAvatarPosition();
            ava_x = (int)(pos.x/25);
            ava_y = (int)(pos.y/25);
        }

        for(Observation o : allobj){
            Vector2d p = o.position;
            int x = (int)(p.x/25);
            int y= (int)(p.y/25);
            map[x][y] = o.itype;
            if(o.itype == 5) {
                //System.out.println((int)(p.x/25)  + " + " + (int)(p.y/25));
                int bomb_x = (int)(p.x/25);
                int bomb_y = (int)(p.y/25);
                if(Math.abs(ava_x-bomb_x)<5 && Math.abs(ava_y-bomb_y)<6)
                    dangerous = true;
            }
        }
        for(int y=0; y<14; y++)
            for(int x=0; x<32; x++)
                feature[y*32+x] = map[x][y];
        
        // 4 states
        feature[448] = obs.getGameTick();
        //feature[449] = obs.getAvatarSpeed();
        //feature[450] = obs.getAvatarHealthPoints();
        //feature[451] = obs.getAvatarType();

        //GameScore
        feature[449] = obs.getGameScore();

        //Position
        feature[450] = ava_x;
        feature[451] = ava_y;

        //Dangerous if there is bomb around the Avatar
        feature[452] = dangerous ? 1 : 0;
        
        return feature;
    }
    
    public static Instances datasetHeader(){
        FastVector attInfo = new FastVector();
        // 448 locations
        for(int y=0; y<14; y++){
            for(int x=0; x<32; x++){
                Attribute att = new Attribute("object_at_position_x=" + x + "_y=" + y);
                attInfo.addElement(att);
            }
        }
        Attribute att = new Attribute("GameTick" ); attInfo.addElement(att);
        //att = new Attribute("AvatarSpeed" ); attInfo.addElement(att);
        //att = new Attribute("AvatarHealthPoints" ); attInfo.addElement(att);
        //att = new Attribute("AvatarType" ); attInfo.addElement(att);
        att = new Attribute("GameScore" ); attInfo.addElement(att);
        att = new Attribute("AvatarPosX" ); attInfo.addElement(att);
        att = new Attribute("AvatarPosY" ); attInfo.addElement(att);
        att = new Attribute("Dangerous" ); attInfo.addElement(att);

        //class
        FastVector classes = new FastVector();
        classes.addElement("0");
        classes.addElement("1");
        classes.addElement("2");
        classes.addElement("3");
        att = new Attribute("class", classes);        
        attInfo.addElement(att);

        Instances instances = new Instances("AliensData", attInfo, 0);
        instances.setClassIndex( instances.numAttributes() - 1);
        
        return instances;
    }

    // Record each move as the ARFF instance
    public void invoke(StateObservation obs, Types.ACTIONS action) {
        double[]  feature = featureExtract(obs);
        
        try{  
            for(int i=0; i<feature.length-1; i++)
                filewriter.write(feature[i] + ",");
            // Recorde the move type as ARFF classes
            int action_num = 0;
            if( Types.ACTIONS.ACTION_NIL == action) action_num = 0;
            if( Types.ACTIONS.ACTION_USE == action) action_num = 1;
            if( Types.ACTIONS.ACTION_LEFT == action) action_num = 2;
            if( Types.ACTIONS.ACTION_RIGHT == action) action_num = 3;
            filewriter.write(action_num + "\n");
            filewriter.flush();
        }catch(Exception exc){
            exc.printStackTrace();
        }
    }
    
    public void close(){
        try{
            filewriter.close();
        }catch(Exception exc){
            exc.printStackTrace();
        }
    }
}
