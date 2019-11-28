package controllers.limitdepthfirst;

import java.util.ArrayList;
import tools.Vector2d;

import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;

public class Agent extends AbstractPlayer {
    protected ArrayList<StateObservation> flag = new ArrayList<StateObservation>();
    public Vector2d keypos, goalpos, box1, box2;
    public int DEPTH;
    public int best_f;
    public Types.ACTIONS best_action, now_action;
    public int is_get_key;

    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer)
    {
        ArrayList<Observation>[] movingPositions = so.getMovablePositions();
        ArrayList<Observation>[] fixedPositions = so.getImmovablePositions();
        keypos = movingPositions[0].get(0).position;//钥匙的坐标
        goalpos = fixedPositions[1].get(0).position; //目标的坐标
        best_f = Integer.MAX_VALUE;
        DEPTH = 3;
        is_get_key = 0;
    }

    public int h(StateObservation state)
    {
        Vector2d now = state.getAvatarPosition();

        if(is_get_key == 1) //得到钥匙
        {
            return l1_norm(now, goalpos);
        }
        else
        {
            if(state.getAvatarType() == 4)
            {
                best_action = now_action;
                best_f = (int) (Math.abs(now.x - keypos.x) + Math.abs(now.y - keypos.y));
                return l1_norm(now, keypos);
            }

            ArrayList<Observation>[] movingPositions = state.getMovablePositions();
            box1 = movingPositions[1].get(0).position;
            box2 = movingPositions[1].get(1).position;
            if(box1.equals(keypos) || box2.equals(keypos))
                return Integer.MAX_VALUE;

            return  l1_norm(now, keypos);
        }
    }

    public void limitDFS(StateObservation stateObs, int depth)
    {
        Types.ACTIONS action = null;
        ArrayList<Types.ACTIONS> actions = stateObs.getAvailableActions();

        if(stateObs.getGameWinner() == Types.WINNER.PLAYER_WINS)
        {
            best_action = now_action;
            best_f = 0;
            return;
        }
        else if(stateObs.isGameOver() && stateObs.getGameWinner() == Types.WINNER.PLAYER_LOSES)
        {
            return;
        }

        if(depth > DEPTH)
        {
            if(h(stateObs) < best_f)
            {
                best_action = now_action;
                best_f = h(stateObs);
            }
            return;
        }

        for(int i = 0; i < actions.size(); i++)
        {
            StateObservation stCopy = stateObs.copy();
            action = actions.get(i);
            stCopy.advance(action);

            if(depth == 1)//记录第一个动作
            {
                now_action = action;
                flag.clear();
                flag.add(stateObs);
            }

            if(isVisited(stCopy))
                continue;

            flag.add(stCopy);
            limitDFS(stCopy, depth+1);
        }
        return;
    }

    public boolean isVisited(StateObservation stateObs)
    {
        for(StateObservation state: flag)
            if(state.equalPosition(stateObs))
                return true;
        return false;
    }

    public int l1_norm(Vector2d a, Vector2d b)
    {
        return (int)(Math.abs(a.x-b.x)+Math.abs(a.y-b.y));
    }

    //0 left  1 right  2 down  3 up

    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
        if(stateObs.getAvatarType() == 4)
        {
            is_get_key = 1;
            best_f = Integer.MAX_VALUE;
        }

        limitDFS(stateObs, 1);

        try{
            Thread.sleep(50);
        }catch(Exception e){
            System.exit(0);
        }

        return best_action;
    }
}


