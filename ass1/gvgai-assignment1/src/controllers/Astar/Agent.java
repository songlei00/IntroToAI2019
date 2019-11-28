package controllers.Astar;

import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;

import java.util.PriorityQueue;
import java.util.ArrayList;
import tools.Vector2d;

public class Agent extends AbstractPlayer {
    public Vector2d keypos, goalpos;
    public static PriorityQueue<Node> closedList, openList;
    public Node best_state;
    public int best_f;
    public int is_get_key;

    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer)
    {
        ArrayList<Observation>[] movingPositions = so.getMovablePositions();
        ArrayList<Observation>[] fixedPositions = so.getImmovablePositions();
        keypos = movingPositions[0].get(0).position;//钥匙的坐标
        goalpos = fixedPositions[1].get(0).position; //目标的坐标

        openList = new PriorityQueue<Node>();
        closedList = new PriorityQueue<Node>();

        is_get_key = 0;
    }

    public int get_h(StateObservation state)
    {
        if(state.getGameWinner() == Types.WINNER.PLAYER_LOSES)
            return 100000;
        if(state.getGameWinner() == Types.WINNER.PLAYER_WINS)
            return 0;

        Vector2d nowpos = state.getAvatarPosition();
        if(is_get_key == 1)
            return l1_norm(nowpos, goalpos);
        else
        {
            ArrayList<Observation>[] fixedPositions = state.getImmovablePositions();
            ArrayList<Observation>[] movingPositions = state.getMovablePositions();

            ArrayList<Observation> holes = null, boxes = null;
            if (fixedPositions.length > 2)
                holes = fixedPositions[fixedPositions.length - 2];
            if (movingPositions != null)
                boxes = movingPositions[movingPositions.length - 1];

            //打印
            /*if(boxes != null && boxes.size() > 0)
                System.out.println("box:" + boxes.get(0).position);
            if(holes != null && holes.size() > 0)
                System.out.println("holes" + holes.get(0).position);*/
            int h = 0;
            //计算箱子和洞的距离
            if (boxes != null && holes != null && boxes.size() > 0 && holes.size() > 0)
                //这里有一个问题 这样精灵会过于倾向于比较靠近的箱子填最近的洞 怎么只判断和钥匙离得近的洞并填上
                //通过减去填的洞的数量乘一个比例可以减少填洞，但这样只能过三关 第四关依然用过多箱子填洞，而最后没有箱子填出通往钥匙的路
            {
                for(Observation box: boxes)
                {
                    for(Observation hole: holes)
                    {
                        h += l1_norm(box.position, hole.position);
                    }
                }
            }

            return h+l1_norm(nowpos, keypos)-(int)(state.getGameScore())*50;
            //第五关时使用下面这个返回值 并且延长时间 可以通过
            //return h+l1_norm(nowpos, keypos)+(int)(state.getGameScore())*50;
        }
    }

    public void Astar(StateObservation stateObs, ElapsedCpuTimer elapsedTimer)
    {
        Types.ACTIONS action = null;
        ArrayList<Types.ACTIONS> actions = stateObs.getAvailableActions();

        //初始化
        openList.clear();
        closedList.clear();
        best_state = null;
        best_f = Integer.MAX_VALUE;

        Node tmp = new Node(stateObs);
        tmp.parent = null;
        tmp.g = 0;
        tmp.h = get_h(stateObs);
        tmp.update_f();
        tmp.stateObs = stateObs;
        openList.add(tmp);

        best_state = tmp;
        best_f = tmp.f;

        int remainingLimit = 20;//这个值应该随着算法改变 算法越复杂 这个应该调大
        while(!openList.isEmpty() && elapsedTimer.remainingTimeMillis()>remainingLimit)
        {
            Node node = openList.poll();
            closedList.add(node);

            for(int i = 0; i < actions.size(); i++)
            {
                StateObservation stCopy = node.stateObs.copy();
                action = actions.get(i);
                stCopy.advance(action);

                if(stCopy.equalPosition(node.stateObs) || isInColse(stCopy))
                    continue;

                Node now = new Node(stCopy);
                now.g = node.g + 1;
                now.parent = node;

                if(stCopy.getGameWinner() == Types.WINNER.PLAYER_WINS)
                {
                    best_state = now;
                    return;
                }
                else if(stCopy.isGameOver() && stCopy.getGameWinner() == Types.WINNER.PLAYER_LOSES)
                {
                    continue;
                }

                Node pre = isInOpen(now);

                if((pre != null && now.g+get_h(now.stateObs) < pre.f) || pre == null)
                {
                    now.h = get_h(now.stateObs);
                    now.update_f();
                    openList.add(now);
                }
                if(now.f < best_f)
                    best_state = now;
            }
        }
    }

    private Node isInOpen(Node now)
    {
        for(Node node: openList)
            if(node.stateObs.equalPosition(now.stateObs))
                return node;
        return null;
    }

    private boolean isInColse(StateObservation now)
    {
        for(Node node: closedList)
            if(node.stateObs.equalPosition(now))
                return true;
        return false;
    }

    public Types.ACTIONS getAction()
    {
        Node p = best_state;
        while(p!= null && p.parent != null && p.parent.parent != null)
        {
            p = p.parent;
        }
        if(p != null)
            return p.stateObs.getAvatarLastAction();
        return null;
    }

    public int l1_norm(Vector2d a, Vector2d b)
    {
        return (int)(Math.abs(a.x-b.x)+Math.abs(a.y-b.y));
    }

    //0 left  1 right  2 down  3 up

    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
        Types.ACTIONS action = null;

        if(stateObs.getAvatarType() == 4)//拿到钥匙后对之前的状态进行恢复
        {
            is_get_key = 1;
            best_f = Integer.MAX_VALUE;
        }

        Astar(stateObs, elapsedTimer);
        action = getAction();
        System.out.println(action);

        try{
            Thread.sleep(50);
        }catch(Exception e){
            System.exit(0);
        }

        return action;
    }
}





