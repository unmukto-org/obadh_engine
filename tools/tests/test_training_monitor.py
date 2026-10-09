import json
from pathlib import Path
import tempfile
import unittest
from tools.correction.training_monitor import run, update_progress, parse_steps, notification_due


class TrainingMonitorTests(unittest.TestCase):
    def config(self, **kwargs):
        return dict(allocation_end_epoch=10000, stall_seconds=900, pipeline_status_files=[], **kwargs)

    def test_running_progress_then_stall_then_recovery(self):
        log=json.dumps(dict(event='byte_train',step=500))
        first=update_progress({},'1.2','obadh-train',log,1000,900)
        second=update_progress({'1.2':first},'1.2','obadh-train',log,1900,900)
        self.assertTrue(second['stalled'])
        third=update_progress({'1.2':second},'1.2','obadh-train',json.dumps(dict(event='byte_train',step=525)),1901,900)
        self.assertFalse(third['stalled'])
        # A new Slurm step using the same name starts a new progress clock.
        restarted=update_progress({'1.2':second},'1.3','obadh-train',log,3000,900)
        self.assertFalse(restarted['stalled'])

    def test_complete_idle_has_heartbeat_and_alert_with_throttling(self):
        result=run(self.config(),{},1000,lambda name: '' if name=='ps' else 'GPU 0 percent')
        self.assertEqual(result['status'],'completed_idle')
        self.assertEqual(result['checked_epoch'],1000)
        self.assertTrue(notification_due(result,{},1000,1800))
        state=dict(result,last_notification_epoch=1000)
        self.assertFalse(notification_due(result,state,1100,1800))
        self.assertTrue(notification_due(result,state,2800,1800))

    def test_missing_ssh_latches_and_never_retries_authentication(self):
        calls=[]
        def failed(*args):
            calls.append(args);raise RuntimeError('no ssh master at socket')
        first=run(self.config(),{},1000,failed)
        self.assertEqual(first['status'],'connection_required')
        self.assertEqual(len(calls),1)
        second=run(self.config(),first,1100,failed)
        self.assertEqual(len(calls),1)
        self.assertTrue(second['connection_latched'])

    def test_pipeline_failure_and_orchestration_stall_are_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'pipeline.json';config=self.config();config['pipeline_status_files']=[str(path)]
            path.write_text(json.dumps(dict(status='blocked')))
            result=run(config,{},1000,lambda *args:'')
            self.assertEqual(result['status'],'pipeline_failed')
            path.write_text(json.dumps(dict(status='training')))
            first=run(config,{},1000,lambda *args:'')
            self.assertEqual(first['status'],'between_stages')
            second=run(config,first,1900,lambda *args:'')
            self.assertEqual(second['status'],'orchestration_stalled')
            path.write_text(json.dumps(dict(status='complete')))
            third=run(config,second,2000,lambda *args:'')
            self.assertEqual(third['status'],'completed_idle')

    def test_active_progress_only_calls_read_operations(self):
        calls=[]
        def remote(*args):
            calls.append(args)
            return {'ps':'3314835.153 obadh-training 2:00\n','logs':json.dumps(dict(event='byte_train',step=1000)),'info':'GPU 96 percent'}[args[0]]
        result=run(self.config(),{},1000,remote)
        self.assertEqual(result['status'],'running')
        self.assertEqual(result['job_progress']['3314835.153']['progress']['step'],1000)
        self.assertEqual({c[0] for c in calls},{'ps','logs','info'})

    def test_deadline_stops_remote_polling_and_bad_listing_fails_closed(self):
        def unexpected(*args):raise AssertionError('must not poll after deadline')
        self.assertEqual(run(self.config(),{},10000,unexpected)['status'],'deadline_reached')
        with self.assertRaises(ValueError):parse_steps('unexpected warning')
        result=run(self.config(),{},1000,lambda *args:'unexpected warning')
        self.assertEqual(result['status'],'monitor_error')
