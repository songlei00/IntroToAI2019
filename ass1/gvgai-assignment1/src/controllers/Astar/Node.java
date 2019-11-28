package controllers.Astar;

import core.game.StateObservation;

public class Node implements Comparable<Node> {
    public int g;
    public int h;
    public int f;
    public Node parent;
    public StateObservation stateObs;

    public Node(StateObservation state)
    {
        stateObs = state.copy();
    }

    public void update_f()
    {
        this.f = this.g + this.h;
    }

    @Override
    public int compareTo(Node n) {
        return (int)(this.f - n.f);
    }
}
