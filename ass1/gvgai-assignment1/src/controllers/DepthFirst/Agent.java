package controllers.DepthFirst;

import java.util.ArrayList;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;

public class Agent extends AbstractPlayer {
    protected ArrayList<StateObservation> flag = new ArrayList<StateObservation>();
    protected ArrayList<Types.ACTIONS> ans = new ArrayList<Types.ACTIONS>();

    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer)
    {
        DFS(so);
    }

    public boolean DFS(StateObservation stateObs)
    {
        Types.ACTIONS action = null;
        ArrayList<Types.ACTIONS> actions = stateObs.getAvailableActions();

        if(stateObs.getGameWinner() == Types.WINNER.PLAYER_WINS)
        {
            System.out.println("Get the path.");
            return true;
        }
        else if(stateObs.isGameOver() && stateObs.getGameWinner() == Types.WINNER.PLAYER_LOSES)
        {
            return false;
        }

        for(int i = 0; i < actions.size(); i++)
        {
            StateObservation stCopy = stateObs.copy();
            action = actions.get(i);
            stCopy.advance(action);

            if(isVisited(stCopy))
                continue;
            ans.add(action);
            flag.add(stCopy);
            if(DFS(stCopy))
                return true;
            else
                ans.remove(ans.size()-1);
        }
        return false;
    }

    public boolean isVisited(StateObservation stateObs)
    {
        for(StateObservation state: flag)
            if(state.equalPosition(stateObs))
                return true;
        return false;
    }

    //0 left  1 right  2 down  3 up
    public int cnt = 0;

    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
        Types.ACTIONS action = null;
        ArrayList<Types.ACTIONS> actions = stateObs.getAvailableActions();

        action = ans.get(cnt);
        cnt++;
        try{
            Thread.sleep(50);
        }catch(Exception e){
            System.exit(0);
        }

        return action;
    }
}


